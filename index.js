#!/usr/bin/env node
import Fastify from 'fastify'
import { TextDecoder } from 'util'

// Base URL: point this at vLLM or any OpenAI-compatible backend.
// e.g. export ANTHROPIC_PROXY_BASE_URL="http://192.168.1.111:8001"
const baseUrl = process.env.ANTHROPIC_PROXY_BASE_URL || 'https://openrouter.ai/api'
const requiresApiKey = !process.env.ANTHROPIC_PROXY_BASE_URL
const key = requiresApiKey ? process.env.OPENROUTER_API_KEY : null

// Default upstream model when using OpenRouter fallback.
// Ignored when ANTHROPIC_PROXY_BASE_URL points at vLLM.
const model = 'google/gemini-2.0-pro-exp-02-05:free'
const models = {
  reasoning: process.env.REASONING_MODEL || model,
  completion: process.env.COMPLETION_MODEL || model,
}

const fastify = Fastify({ logger: true })

function debug(...args) {
  if (!process.env.DEBUG) return
  console.log(...args)
}

// Helper to send SSE quickly
const sendSSE = (reply, event, data) => {
  const sseMessage = `event: ${event}\n` + `data: ${JSON.stringify(data)}\n\n`
  reply.raw.write(sseMessage)
  if (typeof reply.raw.flush === 'function') reply.raw.flush()
}

function mapStopReason(finishReason) {
  switch (finishReason) {
    case 'tool_calls': return 'tool_use'
    case 'stop': return 'end_turn'
    case 'length': return 'max_tokens'
    default: return 'end_turn'
  }
}

// Normalize only text blocks from Anthropic content arrays
function extractTextFromBlocks(content) {
  if (typeof content === 'string') return content
  if (Array.isArray(content)) {
    return content
      .filter(b => b && b.type === 'text' && typeof b.text === 'string')
      .map(b => b.text)
      .join(' ')
  }
  return null
}

// Convert Anthropic tool_result.content → string
function stringifyToolResultContent(block) {
  // Anthropic tool_result content can be:
  // - string: "..."
  // - object block { type: "json", json: {...} }
  // - arbitrary object
  if (typeof block === 'string') return block
  if (block && typeof block === 'object') {
    if (block.type === 'json' && block.json !== undefined) {
      try { return JSON.stringify(block.json) } catch { /* fallthrough */ }
    }
    try { return JSON.stringify(block) } catch { return String(block) }
  }
  return ''
}

fastify.post('/v1/messages', async (request, reply) => {
  try {
    const payload = request.body || {}

    // We keep a mapping so we can include function `name` on tool replies.
    // key: tool_use_id (OpenAI tool_call id), value: function name
    const toolNameById = new Map()

    // Build OpenAI ChatCompletions-style messages array
    const messages = []

    // Anthropic "system" can be string or array of system blocks; support both.
    if (payload.system) {
      if (Array.isArray(payload.system)) {
        payload.system.forEach(sysMsg => {
          const txt = extractTextFromBlocks(sysMsg?.text ?? sysMsg?.content ?? sysMsg)
          if (txt) messages.push({ role: 'system', content: txt })
        })
      } else {
        const txt = typeof payload.system === 'string'
          ? payload.system
          : extractTextFromBlocks(payload.system)
        if (txt) messages.push({ role: 'system', content: txt })
      }
    }

    if (Array.isArray(payload.messages)) {
      payload.messages.forEach(msg => {
        const role = msg.role // 'user' | 'assistant'
        const textContent = extractTextFromBlocks(msg.content)

        // Gather tool_use blocks (Anthropic assistant content)
        const contentArray = Array.isArray(msg.content) ? msg.content : []
        const toolUses = contentArray.filter(b => b && b.type === 'tool_use')
        const toolResults = contentArray.filter(b => b && b.type === 'tool_result')

        // If there are tool_use blocks, we must emit an OpenAI assistant message
        // with `tool_calls` and (optionally) text content.
        if (toolUses.length > 0) {
          const tool_calls = toolUses.map((tu) => {
            const id = String(tu.id || '').trim()
            const name = String(tu.name || '').trim()
            const inputObj = tu.input ?? {}

            // Remember mapping for later tool_result → OpenAI tool message
            if (id && name) toolNameById.set(id, name)

            return {
              id,                                // MUST be 9-char alnum if coming from vLLM; keep as-is
              type: 'function',
              function: {
                name,
                // MUST be a JSON STRING, not object
                arguments: JSON.stringify(inputObj)
              }
            }
          })

          const assistantMsg = {
            role: 'assistant',
            content: textContent || '', // OpenAI requires string (can be '')
            tool_calls
          }
          messages.push(assistantMsg)
        } else if (role === 'assistant' || role === 'user') {
          // Regular assistant/user message (no tool_use blocks)
          const newMsg = { role, content: textContent || '' }
          if (newMsg.content) messages.push(newMsg)
        }

        // Emit OpenAI tool messages for each tool_result block
        // We must include tool_call_id and (ideally) the function name.
        toolResults.forEach(tr => {
          const tool_call_id = String(tr.tool_use_id || '').trim()
          const name = toolNameById.get(tool_call_id) || tr.name || 'tool'
          const content = stringifyToolResultContent(tr.content ?? tr.text)
          messages.push({
            role: 'tool',
            tool_call_id,
            name,
            content
          })
        })
      })
    }

    // Tools → OpenAI function tools
    const removeUriFormat = (schema) => {
      if (!schema || typeof schema !== 'object') return schema

      if (schema.type === 'string' && schema.format === 'uri') {
        const { format, ...rest } = schema
        return rest
      }

      if (Array.isArray(schema)) return schema.map(removeUriFormat)

      const result = {}
      for (const k in schema) {
        if (k === 'properties' && typeof schema[k] === 'object') {
          result[k] = {}
          for (const pk in schema[k]) {
            result[k][pk] = removeUriFormat(schema[k][pk])
          }
        } else if (k === 'items' && typeof schema[k] === 'object') {
          result[k] = removeUriFormat(schema[k])
        } else if (k === 'additionalProperties' && typeof schema[k] === 'object') {
          result[k] = removeUriFormat(schema[k])
        } else if (['anyOf', 'allOf', 'oneOf'].includes(k) && Array.isArray(schema[k])) {
          result[k] = schema[k].map(removeUriFormat)
        } else {
          result[k] = removeUriFormat(schema[k])
        }
      }
      return result
    }

    const tools = (payload.tools || [])
      .filter(tool => !['BatchTool'].includes(tool.name))
      .map(tool => ({
        type: 'function',
        function: {
          name: tool.name,
          description: tool.description,
          parameters: removeUriFormat(tool.input_schema),
        }
      }))

    const openaiPayload = {
      model: payload.thinking ? models.reasoning : models.completion,
      messages,
      max_tokens: payload.max_tokens,
      temperature: payload.temperature !== undefined ? payload.temperature : 1,
      stream: payload.stream === true,
    }
    if (tools.length > 0) openaiPayload.tools = tools

    debug('OpenAI payload:', openaiPayload)

    const headers = { 'Content-Type': 'application/json' }
    if (requiresApiKey) headers['Authorization'] = `Bearer ${key}`

    const openaiResponse = await fetch(`${baseUrl}/v1/chat/completions`, {
      method: 'POST',
      headers,
      body: JSON.stringify(openaiPayload)
    })

    if (!openaiResponse.ok) {
      const errorDetails = await openaiResponse.text()
      reply.code(openaiResponse.status)
      return { error: errorDetails }
    }

    // Non-streaming
    if (!openaiPayload.stream) {
      const data = await openaiResponse.json()
      debug('OpenAI response:', data)
      if (data.error) throw new Error(data.error.message)

      const choice = data.choices[0]
      const openaiMessage = choice.message || {}
      const stopReason = mapStopReason(choice.finish_reason)
      const toolCalls = openaiMessage.tool_calls || []

      const messageId = data.id
        ? data.id.replace('chatcmpl', 'msg')
        : 'msg_' + Math.random().toString(36).slice(2, 26)

      const anthropicResponse = {
        content: [
          ...(openaiMessage.content
            ? [{ type: 'text', text: openaiMessage.content }]
            : []),
          ...toolCalls.map(tc => ({
            type: 'tool_use',
            id: tc.id,
            name: tc.function?.name,
            input: (() => {
              try { return JSON.parse(tc.function?.arguments || '{}') }
              catch { return {} }
            })()
          }))
        ],
        id: messageId,
        model: openaiPayload.model,
        role: openaiMessage.role || 'assistant',
        stop_reason: stopReason,
        stop_sequence: null,
        type: 'message',
        usage: {
          input_tokens: data.usage?.prompt_tokens ?? 0,
          output_tokens: data.usage?.completion_tokens ?? 0,
        }
      }

      return anthropicResponse
    }

    // Streaming path (OpenAI → Anthropic SSE)
    let isSucceeded = false
    function sendSuccessMessage() {
      if (isSucceeded) return
      isSucceeded = true
      reply.raw.writeHead(200, {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        Connection: 'keep-alive'
      })
      const messageId = 'msg_' + Math.random().toString(36).slice(2, 26)
      sendSSE(reply, 'message_start', {
        type: 'message_start',
        message: {
          id: messageId,
          type: 'message',
          role: 'assistant',
          model: openaiPayload.model,
          content: [],
          stop_reason: null,
          stop_sequence: null,
          usage: { input_tokens: 0, output_tokens: 0 }
        }
      })
      sendSSE(reply, 'ping', { type: 'ping' })
    }

    let accumulatedContent = ''
    let accumulatedReasoning = ''
    let usage = null
    let textBlockStarted = false
    let encounteredToolCall = false
    const toolCallAccumulators = {}  // index -> accumulated arguments
    const decoder = new TextDecoder('utf-8')
    const reader = openaiResponse.body.getReader()
    let done = false

    while (!done) {
      const { value, done: doneReading } = await reader.read()
      done = doneReading
      if (!value) continue

      const chunk = decoder.decode(value)
      debug('OpenAI response chunk:', chunk)
      const lines = chunk.split('\n')

      for (const line of lines) {
        const trimmed = line.trim()
        if (!trimmed || !trimmed.startsWith('data:')) continue
        const dataStr = trimmed.replace(/^data:\s*/, '')
        if (dataStr === '[DONE]') {
          if (encounteredToolCall) {
            for (const idx in toolCallAccumulators) {
              sendSSE(reply, 'content_block_stop', {
                type: 'content_block_stop',
                index: parseInt(idx, 10)
              })
            }
          } else if (textBlockStarted) {
            sendSSE(reply, 'content_block_stop', {
              type: 'content_block_stop',
              index: 0
            })
          }
          sendSSE(reply, 'message_delta', {
            type: 'message_delta',
            delta: {
              stop_reason: encounteredToolCall ? 'tool_use' : 'end_turn',
              stop_sequence: null
            },
            usage: usage
              ? { output_tokens: usage.completion_tokens }
              : { output_tokens: (accumulatedContent.split(/\s+/).filter(Boolean).length +
                                  accumulatedReasoning.split(/\s+/).filter(Boolean).length) }
          })
          sendSSE(reply, 'message_stop', { type: 'message_stop' })
          reply.raw.end()
          return
        }

        const parsed = JSON.parse(dataStr)
        if (parsed.error) throw new Error(parsed.error.message)
        sendSuccessMessage()

        if (parsed.usage) usage = parsed.usage
        const delta = parsed.choices?.[0]?.delta || {}

        if (delta.tool_calls) {
          for (const tc of delta.tool_calls) {
            encounteredToolCall = true
            const idx = tc.index ?? 0
            if (toolCallAccumulators[idx] === undefined) {
              toolCallAccumulators[idx] = ''
              sendSSE(reply, 'content_block_start', {
                type: 'content_block_start',
                index: idx,
                content_block: {
                  type: 'tool_use',
                  id: tc.id,
                  name: tc.function?.name,
                  input: {}
                }
              })
            }
            const newArgs = tc.function?.arguments || ''
            const oldArgs = toolCallAccumulators[idx]
            if (newArgs.length > oldArgs.length) {
              const deltaText = newArgs.substring(oldArgs.length)
              sendSSE(reply, 'content_block_delta', {
                type: 'content_block_delta',
                index: idx,
                delta: { type: 'input_json_delta', partial_json: deltaText }
              })
              toolCallAccumulators[idx] = newArgs
            }
          }
        } else if (typeof delta.content === 'string' && delta.content.length) {
          if (!textBlockStarted) {
            textBlockStarted = true
            sendSSE(reply, 'content_block_start', {
              type: 'content_block_start',
              index: 0,
              content_block: { type: 'text', text: '' }
            })
          }
          accumulatedContent += delta.content
          sendSSE(reply, 'content_block_delta', {
            type: 'content_block_delta',
            index: 0,
            delta: { type: 'text_delta', text: delta.content }
          })
        } else if (typeof delta.reasoning === 'string' && delta.reasoning.length) {
          if (!textBlockStarted) {
            textBlockStarted = true
            sendSSE(reply, 'content_block_start', {
              type: 'content_block_start',
              index: 0,
              content_block: { type: 'text', text: '' }
            })
          }
          accumulatedReasoning += delta.reasoning
          sendSSE(reply, 'content_block_delta', {
            type: 'content_block_delta',
            index: 0,
            delta: { type: 'thinking_delta', thinking: delta.reasoning }
          })
        }
      }
    }

    reply.raw.end()
  } catch (err) {
    console.error(err)
    reply.code(500)
    return { error: err?.message || String(err) }
  }
})

const start = async () => {
  try {
    await fastify.listen({ port: process.env.PORT || 3000 })
  } catch (err) {
    process.exit(1)
  }
}

start()
