import { useState, useRef, useEffect } from 'react'
import { Send, Loader, Network, Zap, FileText, ChevronDown, ChevronUp,
         Copy, CheckCheck, Shuffle, Terminal } from 'lucide-react'
import { runQuery } from '../hooks/useResults'
import NetworkViz  from '../components/NetworkViz'
import AgenticViz  from '../components/AgenticViz'
import RAGViz      from '../components/RAGViz'

const PIPELINES = [
  { id: 'graphrag', label: 'GraphRAG',        accent: '#0091ff', icon: Network,   desc: '0 LLM tokens' },
  { id: 'agentic',  label: 'Agentic',         accent: '#a855f7', icon: Zap,       desc: 'LLM planning' },
  { id: 'rag',      label: 'RAG',             accent: '#f97316', icon: FileText,  desc: 'Vector top-5' },
]

const EXAMPLES = [
  'How many athletics events had more than 50 competitors at the 2012 Summer Olympics?',
  'Which shooting event at the 2008 Summer Olympics had the most competitors?',
  'Who won the gold medal in cross-country skiing immediately before the 2010 Winter Olympics?',
  'How many nations competed in Athletics at the 2016 Summer Olympics – Men\'s 100 metres?',
  'Summarise the athletics events at the 2012 Summer Olympics',
]

// ── Trace Panel ────────────────────────────────────────────────────────────────
const OP_STYLE = {
  SOURCE:  { color: '#0091ff', bg: '#0091ff18', label: 'SOURCE'  },
  FILTER:  { color: '#10b981', bg: '#10b98118', label: 'FILTER'  },
  COUNT:   { color: '#3b82f6', bg: '#3b82f618', label: 'COUNT'   },
  SORT:    { color: '#f59e0b', bg: '#f59e0b18', label: 'SORT'    },
  EDGE:    { color: '#06b6d4', bg: '#06b6d418', label: 'EDGE'    },
  MATCH:   { color: '#8b5cf6', bg: '#8b5cf618', label: 'MATCH'   },
  INFOBOX: { color: '#a78bfa', bg: '#a78bfa18', label: 'INFOBOX' },
  VECTOR:  { color: '#f97316', bg: '#f9731618', label: 'VECTOR'  },
  BUILD:   { color: '#84cc16', bg: '#84cc1618', label: 'BUILD'   },
  FALLBACK:{ color: '#f97316', bg: '#f9731618', label: 'FALLBACK'},
  RETURN:  { color: '#22c55e', bg: '#22c55e18', label: 'RETURN'  },
  STEP:    { color: '#6b7280', bg: '#6b728018', label: 'STEP'    },
}

function TracePanel({ trace }) {
  const [open, setOpen] = useState(false)
  if (!trace?.length) return null
  return (
    <div className="mt-2 rounded-lg border border-surface-600 overflow-hidden">
      <button
        onClick={() => setOpen(o => !o)}
        className="w-full flex items-center justify-between px-3 py-2 bg-surface-800 hover:bg-surface-700 transition-colors text-xs"
      >
        <span className="flex items-center gap-2 text-gray-400 font-semibold">
          <Terminal size={11} />
          Graph Query Trace
          <span className="text-[10px] text-gray-600 font-normal">· {trace.length} steps</span>
        </span>
        {open ? <ChevronUp size={12} className="text-gray-500" /> : <ChevronDown size={12} className="text-gray-500" />}
      </button>
      {open && (
        <div className="bg-[#050c18] p-2 space-y-1 font-mono text-[10px] max-h-52 overflow-y-auto">
          {trace.map((step, i) => {
            const s = OP_STYLE[step.op] || OP_STYLE.STEP
            return (
              <div key={i} className="flex gap-2 items-start">
                <span className="flex-shrink-0 px-1.5 py-0.5 rounded text-[9px] font-bold"
                      style={{ background: s.bg, color: s.color }}>
                  {s.label}
                </span>
                <div className="flex-1 min-w-0">
                  <p className="text-gray-300 leading-relaxed break-all">{step.detail}</p>
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
  )
}

// ── Speed Race Bar ─────────────────────────────────────────────────────────────
// Log scale so bars always grow — no cross-pipeline dependency, no backward motion
const LOG_CAP = 200  // seconds at 100% width

function SpeedRace({ elapsed, loading }) {
  return (
    <div className="bg-surface-800 border border-surface-600 rounded-xl px-5 py-4">
      <p className="text-[10px] text-gray-500 uppercase tracking-widest mb-3">Pipeline Speed Race</p>
      <div className="space-y-2.5">
        {PIPELINES.map(p => {
          const t = elapsed[p.id]
          const isRunning = loading[p.id]
          // Each bar grows independently on a log scale — never shrinks
          const pct = t
            ? Math.min(Math.log1p(t) / Math.log1p(LOG_CAP) * 100, 100)
            : 0
          return (
            <div key={p.id} className="flex items-center gap-3">
              <span className="text-xs font-semibold w-20 text-right" style={{ color: p.accent }}>
                {p.label}
              </span>
              <div className="flex-1 h-2.5 bg-surface-700 rounded-full overflow-hidden">
                <div
                  className="h-full rounded-full transition-all duration-200"
                  style={{
                    width: `${pct}%`,
                    background: `linear-gradient(90deg, ${p.accent}60, ${p.accent})`,
                    boxShadow: isRunning && pct > 0 ? `0 0 8px ${p.accent}60` : 'none',
                  }}
                />
              </div>
              <span className="text-xs font-mono w-14 text-left" style={{ color: p.accent }}>
                {isRunning ? (
                  <span className="animate-pulse text-gray-500">{t ? `${t.toFixed(1)}s` : '…'}</span>
                ) : t ? (
                  <span className="font-bold">{t.toFixed(2)}s</span>
                ) : (
                  <span className="text-gray-700">—</span>
                )}
              </span>
            </div>
          )
        })}
      </div>
    </div>
  )
}

// ── Column ─────────────────────────────────────────────────────────────────────
function PipelineColumn({ pipeline, result, loading, animDone, onAnimDone }) {
  const { id, label, accent, icon: Icon, desc } = pipeline
  const isDone    = !!result && !loading
  const isRunning = loading

  const [copied, setCopied] = useState(false)
  function copyAnswer() {
    navigator.clipboard.writeText(result?.answer || '')
    setCopied(true)
    setTimeout(() => setCopied(false), 1800)
  }

  return (
    <div className="flex flex-col gap-2 min-w-0">
      {/* Column header */}
      <div className="flex items-center justify-between px-1">
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full"
               style={{ background: isDone ? '#22c55e' : isRunning ? accent : '#374151',
                        boxShadow: isRunning ? `0 0 6px ${accent}` : 'none',
                        animation: isRunning ? 'pulse 1s infinite' : 'none' }} />
          <Icon size={13} style={{ color: accent }} />
          <span className="text-sm font-bold text-white">{label}</span>
        </div>
        <span className="text-[10px] text-gray-600">{desc}</span>
      </div>

      {/* Visualization — compact height */}
      {id === 'graphrag' && (
        <NetworkViz
          running={isRunning}
          done={isDone}
          traversedIds={result?.traversed_ids || []}
          traversedNodes={result?.traversed_nodes || []}
          answerIds={result?.answer_node_ids || []}
          onAnimationDone={onAnimDone}
          height={320}
        />
      )}
      {id === 'agentic' && (
        <div className="rounded-xl bg-[#080d14] border border-surface-700 min-h-[320px] max-h-[320px] overflow-y-auto">
          <AgenticViz
            agentSteps={result?.agent_steps || []}
            running={isRunning}
            done={isDone}
          />
        </div>
      )}
      {id === 'rag' && (
        <div className="min-h-[320px]">
          <RAGViz
            docSnippets={result?.doc_snippets || []}
            running={isRunning}
            done={isDone}
          />
        </div>
      )}

      {/* Answer card */}
      {isDone && animDone && result?.answer && (
        <div className="rounded-xl border p-3 transition-all"
             style={{ borderColor: accent + '40', background: accent + '0d' }}>
          <div className="flex items-center justify-between mb-1.5">
            <span className="text-[10px] font-semibold uppercase tracking-wider" style={{ color: accent }}>
              Answer
            </span>
            <div className="flex items-center gap-2">
              {result.elapsed && (
                <span className="text-[10px] text-gray-600 font-mono">{result.elapsed}s</span>
              )}
              {result.tokens > 0
                ? <span className="text-[10px] text-gray-600">{result.tokens} tok</span>
                : <span className="text-[10px] text-green-500">0 tokens</span>
              }
              <button onClick={copyAnswer}
                      className="p-1 rounded hover:bg-surface-600 transition-colors">
                {copied
                  ? <CheckCheck size={11} className="text-green-400" />
                  : <Copy size={11} className="text-gray-500 hover:text-gray-300" />
                }
              </button>
            </div>
          </div>
          <p className={`font-bold text-white leading-snug ${result.answer?.length > 60 ? 'text-sm' : 'text-lg'}`}>
            {result.answer}
          </p>
          {result.used_fallback && (
            <p className="text-[10px] text-orange-400 mt-1">↳ vector fallback</p>
          )}
        </div>
      )}

      {/* Graph query trace — GraphRAG only */}
      {id === 'graphrag' && isDone && (
        <TracePanel trace={result?.query_trace} />
      )}
    </div>
  )
}

// ── Page ───────────────────────────────────────────────────────────────────────
export default function Compare() {
  const [question, setQuestion] = useState('')
  const [loading,  setLoading]  = useState({ graphrag: false, agentic: false, rag: false })
  const [results,  setResults]  = useState({ graphrag: null,  agentic: null,  rag: null  })
  const [elapsed,  setElapsed]  = useState({ graphrag: null,  agentic: null,  rag: null  })
  const [animDone, setAnimDone] = useState({ graphrag: true,  agentic: true,  rag: true  })
  const startRef   = useRef(null)
  const timerRef   = useRef(null)
  const loadingRef = useRef({ graphrag: false, agentic: false, rag: false })

  const anyLoading = Object.values(loading).some(Boolean)

  // Live elapsed ticker — reads loadingRef (always current) not stale closure
  useEffect(() => {
    if (anyLoading && startRef.current) {
      timerRef.current = setInterval(() => {
        const now = (Date.now() - startRef.current) / 1000
        const ldr = loadingRef.current
        setElapsed(prev => ({
          graphrag: ldr.graphrag ? now : prev.graphrag,
          agentic:  ldr.agentic  ? now : prev.agentic,
          rag:      ldr.rag      ? now : prev.rag,
        }))
      }, 80)
    } else {
      clearInterval(timerRef.current)
    }
    return () => clearInterval(timerRef.current)
  }, [anyLoading])

  async function compare(q = question) {
    if (!q.trim() || anyLoading) return
    setQuestion(q)
    setResults({ graphrag: null, agentic: null, rag: null })
    setElapsed({ graphrag: null, agentic: null, rag: null })
    setAnimDone({ graphrag: false, agentic: true, rag: true })
    loadingRef.current = { graphrag: true, agentic: true, rag: true }
    setLoading({ graphrag: true, agentic: true, rag: true })
    startRef.current = Date.now()

    const fire = async (pid) => {
      try {
        const res = await runQuery(q.trim(), pid)
        const t   = (Date.now() - startRef.current) / 1000
        setResults(prev => ({ ...prev, [pid]: res }))
        setElapsed(prev => ({ ...prev, [pid]: parseFloat(t.toFixed(2)) }))
      } catch {
        setResults(prev => ({ ...prev, [pid]: { answer: 'Error', error: true } }))
      } finally {
        setLoading(prev => {
          const next = { ...prev, [pid]: false }
          loadingRef.current = next
          return next
        })
      }
    }

    // All 3 fire simultaneously
    fire('graphrag')
    fire('agentic')
    fire('rag')
  }

  function surprise() {
    const q = EXAMPLES[Math.floor(Math.random() * EXAMPLES.length)]
    setQuestion(q)
    compare(q)
  }

  const hasAnyResult = Object.values(results).some(Boolean)

  return (
    <div className="h-full flex flex-col overflow-hidden">
      {/* Top bar */}
      <div className="flex-shrink-0 px-6 pt-5 pb-4 border-b border-surface-700 bg-surface-800/40 space-y-3">
        <div className="flex items-center justify-between">
          <h1 className="text-xl font-bold text-white">Pipeline Comparison</h1>
          <span className="text-xs text-gray-500">Same question · all 3 pipelines · simultaneously</span>
        </div>
        <div className="flex gap-3">
          <div className="flex-1 relative">
            <textarea
              value={question}
              onChange={e => setQuestion(e.target.value)}
              onKeyDown={e => { if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) compare() }}
              placeholder="Ask an Olympic question to compare all pipelines…"
              rows={2}
              className="w-full px-4 py-3 bg-surface-700 border border-surface-600 rounded-xl text-sm text-gray-200 placeholder-gray-600 focus:outline-none resize-none"
            />
          </div>
          <div className="flex flex-col gap-2 self-stretch">
            <button
              onClick={() => compare()}
              disabled={anyLoading || !question.trim()}
              className="flex-1 px-5 rounded-xl font-semibold text-sm text-white bg-brand-500 hover:bg-brand-400 transition-all disabled:opacity-40 flex items-center gap-2 justify-center"
            >
              {anyLoading
                ? <><Loader size={14} className="animate-spin" /> Running</>
                : <><Send size={14} /> Compare</>}
            </button>
            <button
              onClick={surprise}
              disabled={anyLoading}
              className="flex-1 px-3 rounded-xl text-xs text-gray-400 bg-surface-700 border border-surface-600 hover:border-surface-400 hover:text-gray-200 transition-all disabled:opacity-40 flex items-center gap-1.5 justify-center"
            >
              <Shuffle size={12} /> Surprise me
            </button>
          </div>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto px-6 py-4 space-y-4">
        {/* Speed race — only after first query fires */}
        {(anyLoading || hasAnyResult) && (
          <SpeedRace elapsed={elapsed} loading={loading} />
        )}

        {/* 3-column grid */}
        {(anyLoading || hasAnyResult) ? (
          <div className="grid grid-cols-3 gap-4">
            {PIPELINES.map(p => (
              <PipelineColumn
                key={p.id}
                pipeline={p}
                result={results[p.id]}
                loading={loading[p.id]}
                animDone={animDone[p.id]}
                onAnimDone={() => setAnimDone(prev => ({ ...prev, graphrag: true }))}
              />
            ))}
          </div>
        ) : (
          /* Empty state */
          <div className="flex flex-col items-center justify-center py-20 gap-4">
            <div className="grid grid-cols-3 gap-3 opacity-20">
              {PIPELINES.map(p => (
                <div key={p.id} className="h-32 rounded-xl border"
                     style={{ borderColor: p.accent + '40', background: p.accent + '08' }} />
              ))}
            </div>
            <p className="text-sm text-gray-600 mt-2">Ask a question to compare all three pipelines side by side</p>
            <div className="flex flex-wrap gap-2 justify-center mt-2">
              {EXAMPLES.slice(0, 3).map((ex, i) => (
                <button key={i} onClick={() => { setQuestion(ex); compare(ex) }}
                        className="text-xs text-gray-500 bg-surface-800 border border-surface-700 hover:border-surface-500 hover:text-gray-300 px-3 py-1.5 rounded-lg transition-all max-w-xs text-left truncate">
                  {ex}
                </button>
              ))}
            </div>
          </div>
        )}
      </div>

      <style>{`
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50%       { opacity: 0.6; }
        }
      `}</style>
    </div>
  )
}
