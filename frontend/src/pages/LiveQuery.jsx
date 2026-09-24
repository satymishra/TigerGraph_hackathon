import { useState } from 'react'
import { runQuery } from '../hooks/useResults'
import { Send, Loader, Network, Zap, FileText, ChevronRight,
         Copy, CheckCheck, Shuffle, Terminal, ChevronDown, ChevronUp } from 'lucide-react'
import NetworkViz  from '../components/NetworkViz'
import AgenticViz  from '../components/AgenticViz'
import RAGViz      from '../components/RAGViz'

const EXAMPLES = [
  { q: 'How many athletics events had more than 50 competitors at the 2012 Summer Olympics?',     type: 'aggregation' },
  { q: 'Which shooting event at the 2008 Summer Olympics had the most competitors?',              type: 'superlative' },
  { q: 'Who won the gold medal in cross-country skiing immediately before the 2010 Winter Olympics?', type: 'temporal' },
  { q: 'How many nations competed in Athletics at the 2016 Summer Olympics – Men\'s 100 metres?', type: 'lookup' },
  { q: 'How many swimming events had more than 30 competitors at the 2004 Summer Olympics?',      type: 'aggregation' },
  { q: 'Which aquatics event at the 2008 Summer Olympics had the most competitors?',              type: 'superlative' },
  { q: 'Summarise the athletics events at the 2012 Summer Olympics',                              type: 'summary' },
  { q: 'Give an overview of cycling events at the 2008 Summer Olympics',                          type: 'summary' },
]

const PIPELINES = [
  {
    id: 'graphrag', label: 'GraphRAG', icon: Network, color: 'brand',
    accent: '#0091ff', desc: 'Graph traversal · 0 LLM tokens · fastest',
  },
  {
    id: 'agentic', label: 'Agentic', icon: Zap, color: 'purple',
    accent: '#a855f7', desc: 'LLM plans which tools to call · adaptive',
  },
  {
    id: 'rag', label: 'RAG', icon: FileText, color: 'orange',
    accent: '#f97316', desc: 'Vector similarity search · top-5 docs',
  },
]

const TYPE_COLOR = {
  aggregation: 'text-blue-400',
  lookup:      'text-green-400',
  multi_hop:   'text-purple-400',
  superlative: 'text-yellow-400',
  temporal:    'text-cyan-400',
  summary:     'text-pink-400',
}

export default function LiveQuery() {
  const [question,   setQuestion]  = useState('')
  const [pipeline,   setPipeline]  = useState('graphrag')
  const [loading,    setLoading]   = useState(false)
  const [result,     setResult]    = useState(null)
  const [error,      setError]     = useState(null)
  const [animDone,   setAnimDone]  = useState(true)
  const [copied,     setCopied]    = useState(false)
  const [traceOpen,  setTraceOpen] = useState(false)

  const pipelineMeta = PIPELINES.find(p => p.id === pipeline)

  async function submit(q = question) {
    if (!q.trim() || loading) return
    setLoading(true)
    setResult(null)
    setError(null)
    setCopied(false)
    setTraceOpen(false)
    setAnimDone(pipeline !== 'graphrag')
    try {
      const res = await runQuery(q.trim(), pipeline)
      setResult(res)
    } catch (e) {
      setError(e.message)
      setAnimDone(true)
    } finally {
      setLoading(false)
    }
  }

  function surprise() {
    const q = EXAMPLES[Math.floor(Math.random() * EXAMPLES.length)].q
    setQuestion(q)
    submit(q)
  }

  function copyAnswer() {
    navigator.clipboard.writeText(result?.answer || '')
    setCopied(true)
    setTimeout(() => setCopied(false), 1800)
  }

  const isDone     = !!result && !loading
  const isRunning  = loading
  // Show answer only after animation completes (instant for non-graphrag pipelines)
  const showAnswer = isDone && animDone

  return (
    <div className="h-full flex flex-col overflow-hidden">
      {/* ── Top: input + pipeline selector ── */}
      <div className="flex-shrink-0 px-6 pt-5 pb-4 border-b border-surface-700 bg-surface-800/40 space-y-4">
        <div className="flex items-center justify-between">
          <h1 className="text-xl font-bold text-white">Live Query</h1>
          <span className="text-xs text-gray-500">
            Ask any Olympic question — watch how each pipeline answers it
          </span>
        </div>

        {/* Pipeline tabs */}
        <div className="flex gap-2">
          {PIPELINES.map(p => {
            const Icon = p.icon
            const active = pipeline === p.id
            return (
              <button
                key={p.id}
                onClick={() => { setPipeline(p.id); setResult(null); setError(null) }}
                className={`flex items-center gap-2 px-4 py-2.5 rounded-lg border text-sm font-medium transition-all ${
                  active
                    ? 'text-white border-opacity-50'
                    : 'text-gray-400 border-surface-600 bg-surface-800 hover:text-gray-200'
                }`}
                style={active ? {
                  borderColor: p.accent + '60',
                  background:  p.accent + '15',
                  color:       p.accent,
                } : {}}
              >
                <Icon size={14} />
                {p.label}
              </button>
            )
          })}
        </div>

        {/* Question input */}
        <div className="flex gap-3">
          <div className="flex-1 relative">
            <textarea
              value={question}
              onChange={e => setQuestion(e.target.value)}
              onKeyDown={e => { if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) submit() }}
              placeholder="Ask an Olympic question…"
              rows={2}
              className="w-full px-4 py-3 bg-surface-700 border border-surface-600 rounded-xl text-sm text-gray-200 placeholder-gray-600 focus:outline-none focus:border-opacity-60 resize-none"
              style={{ focusBorderColor: pipelineMeta?.accent }}
            />
          </div>
          <div className="flex flex-col gap-1.5 self-stretch">
            <button
              onClick={() => submit()}
              disabled={loading || !question.trim()}
              className="flex-1 px-5 rounded-xl font-semibold text-sm text-white transition-all disabled:opacity-40 disabled:cursor-not-allowed flex items-center gap-2 justify-center"
              style={{ background: pipelineMeta?.accent || '#0091ff' }}
            >
              {loading
                ? <><Loader size={15} className="animate-spin" /> Running</>
                : <><Send size={15} /> Run</>
              }
            </button>
            <button
              onClick={surprise}
              disabled={loading}
              className="px-3 py-1.5 rounded-xl text-xs text-gray-400 bg-surface-700 border border-surface-600 hover:border-surface-400 hover:text-gray-200 transition-all disabled:opacity-40 flex items-center gap-1.5 justify-center"
            >
              <Shuffle size={11} /> Surprise me
            </button>
          </div>
        </div>
      </div>

      {/* ── Middle: visualization ── */}
      <div className="flex-1 overflow-y-auto px-6 py-4 space-y-4">

        {/* Error */}
        {error && (
          <div className="bg-red-500/10 border border-red-500/30 rounded-xl p-4 text-sm">
            <p className="font-semibold text-red-400 mb-1">Query failed</p>
            <p className="text-red-300/70 text-xs mb-2">{error}</p>
            <p className="text-gray-500 text-xs">
              Make sure backend is running:{' '}
              <code className="bg-surface-700 px-1.5 py-0.5 rounded text-gray-300">
                python -m uvicorn backend.main:app --port 8000
              </code>
            </p>
          </div>
        )}

        {/* GraphRAG — network visualization */}
        {pipeline === 'graphrag' && (
          <div className="space-y-3">
            <div className="flex items-center gap-2">
              <div className="w-1.5 h-1.5 rounded-full bg-brand-500" />
              <span className="text-xs text-gray-400 font-semibold uppercase tracking-wider">
                Graph Traversal Path
              </span>
              {isRunning && <span className="text-xs text-brand-400 animate-pulse">traversing…</span>}
              {isDone && result?.graph_path && (
                <span className="ml-auto text-[10px] text-gray-500 font-mono">
                  {result.elapsed}s · 0 tokens
                </span>
              )}
            </div>
            <NetworkViz
              running={isRunning}
              done={isDone}
              traversedIds={result?.traversed_ids || []}
              traversedNodes={result?.traversed_nodes || []}
              answerIds={result?.answer_node_ids || []}
              onAnimationDone={() => setAnimDone(true)}
            />
          </div>
        )}

        {/* Agentic — step-by-step tool calls */}
        {pipeline === 'agentic' && (
          <AgenticViz
            agentSteps={result?.agent_steps || []}
            running={isRunning}
            done={isDone}
          />
        )}

        {/* RAG — document retrieval */}
        {pipeline === 'rag' && (
          <RAGViz
            docSnippets={result?.doc_snippets || []}
            running={isRunning}
            done={isDone}
          />
        )}

        {/* ── Answer card ── */}
        {showAnswer && result?.answer && (
          <div
            className="rounded-xl border p-5 transition-all"
            style={{
              borderColor: pipelineMeta?.accent + '40',
              background:  pipelineMeta?.accent + '0d',
            }}
          >
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-semibold uppercase tracking-wider"
                    style={{ color: pipelineMeta?.accent }}>
                Answer
              </span>
              <div className="flex items-center gap-3 text-xs text-gray-500">
                {result.qtype && (
                  <span className={TYPE_COLOR[result.qtype] || 'text-gray-400'}>
                    {result.qtype}
                  </span>
                )}
                <span>{result.elapsed}s</span>
                {result.tokens > 0
                  ? <span>{result.tokens} tokens</span>
                  : <span className="text-green-400">0 LLM tokens</span>
                }
                <button onClick={copyAnswer}
                        className="p-1 rounded hover:bg-surface-600 transition-colors">
                  {copied
                    ? <CheckCheck size={12} className="text-green-400" />
                    : <Copy size={12} className="text-gray-500 hover:text-gray-300" />
                  }
                </button>
              </div>
            </div>
            <p className={`font-bold text-white leading-snug ${result.answer?.length > 80 ? 'text-base' : 'text-2xl'}`}
               style={{ animation: 'fadeUp 0.4s ease forwards' }}>
              {result.answer}
            </p>
            {result.used_fallback && (
              <p className="text-xs text-orange-400 mt-1">↳ Used vector search fallback</p>
            )}
          </div>
        )}

        {/* ── Query trace (GraphRAG only) ── */}
        {showAnswer && pipeline === 'graphrag' && result?.query_trace?.length > 0 && (
          <div className="rounded-xl border border-surface-700 overflow-hidden">
            <button
              onClick={() => setTraceOpen(o => !o)}
              className="w-full flex items-center justify-between px-4 py-2.5 bg-surface-800 hover:bg-surface-700 transition-colors text-xs"
            >
              <span className="flex items-center gap-2 text-gray-400 font-semibold">
                <Terminal size={12} />
                Graph Query Trace
                <span className="text-[10px] text-gray-600 font-normal">
                  · {result.query_trace.length} steps · TigerGraph REST++
                </span>
              </span>
              {traceOpen
                ? <ChevronUp size={13} className="text-gray-500" />
                : <ChevronDown size={13} className="text-gray-500" />
              }
            </button>
            {traceOpen && (
              <div className="bg-[#050c18] p-3 space-y-1.5 font-mono text-[11px] max-h-60 overflow-y-auto">
                {result.query_trace.map((step, i) => {
                  const opColors = {
                    SOURCE: '#0091ff', FILTER: '#10b981', COUNT: '#3b82f6',
                    SORT: '#f59e0b', EDGE: '#06b6d4', MATCH: '#8b5cf6',
                    INFOBOX: '#a78bfa', VECTOR: '#f97316', BUILD: '#84cc16',
                    FALLBACK: '#f97316', RETURN: '#22c55e', STEP: '#6b7280',
                  }
                  const color = opColors[step.op] || '#6b7280'
                  return (
                    <div key={i} className="flex gap-2 items-start">
                      <span className="flex-shrink-0 px-1.5 py-0.5 rounded text-[9px] font-bold"
                            style={{ background: color + '20', color }}>
                        {step.op}
                      </span>
                      <div className="flex-1 min-w-0">
                        <p className="text-gray-300 leading-relaxed">{step.detail}</p>
                        {step.endpoint && (
                          <p className="text-gray-600 text-[9px] mt-0.5 break-all">{step.endpoint}</p>
                        )}
                      </div>
                    </div>
                  )
                })}
              </div>
            )}
          </div>
        )}

        {/* ── Example queries ── */}
        {!result && !loading && !error && (
          <div>
            <p className="text-xs text-gray-500 uppercase tracking-wider mb-3">Example queries</p>
            <div className="grid grid-cols-1 gap-2">
              {EXAMPLES.map((ex, i) => (
                <button
                  key={i}
                  onClick={() => { setQuestion(ex.q); submit(ex.q) }}
                  className="w-full text-left bg-surface-800 border border-surface-700 hover:border-surface-500 rounded-xl px-4 py-3 transition-all group"
                >
                  <div className="flex items-start gap-3">
                    <ChevronRight size={13} className="mt-0.5 shrink-0 text-gray-600 group-hover:text-gray-400 transition-colors" />
                    <div>
                      <p className="text-sm text-gray-300 group-hover:text-white transition-colors leading-snug">
                        {ex.q}
                      </p>
                      <span className={`text-[10px] font-semibold mt-1 inline-block ${TYPE_COLOR[ex.type]}`}>
                        {ex.type}
                      </span>
                    </div>
                  </div>
                </button>
              ))}
            </div>
          </div>
        )}
      </div>

      <style>{`
        @keyframes fadeUp {
          from { opacity: 0; transform: translateY(8px); }
          to   { opacity: 1; transform: translateY(0);   }
        }
      `}</style>
    </div>
  )
}
