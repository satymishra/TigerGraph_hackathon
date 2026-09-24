import { useEffect, useState, useRef } from 'react'
import {
  Brain, Filter, ArrowUpDown, ArrowRight,
  BookOpen, Search, RefreshCw, Lightbulb,
  CheckCircle2, Loader2, Cpu, ChevronDown,
} from 'lucide-react'

// ── Tool registry ─────────────────────────────────────────────
const TOOLS = {
  llm_plan:       { icon: Brain,         color: '#a855f7', label: 'LLM Plan',      bg: '#a855f715' },
  graph_filter:   { icon: Filter,        color: '#3b82f6', label: 'Graph Filter',  bg: '#3b82f615' },
  sort_results:   { icon: ArrowUpDown,   color: '#10b981', label: 'Sort Results',  bg: '#10b98115' },
  follow_edge:    { icon: ArrowRight,    color: '#f59e0b', label: 'Edge Hop',      bg: '#f59e0b15' },
  prev_edition:   { icon: ArrowRight,    color: '#f59e0b', label: 'Prev Edition',  bg: '#f59e0b15' },
  infobox_lookup: { icon: BookOpen,      color: '#06b6d4', label: 'Infobox',       bg: '#06b6d415' },
  vector_search:  { icon: Search,        color: '#f97316', label: 'Vector Search', bg: '#f9731615' },
  retry_vector:   { icon: RefreshCw,     color: '#6366f1', label: 'Retry',         bg: '#6366f115' },
  final_fallback: { icon: Lightbulb,     color: '#84cc16', label: 'Fallback',      bg: '#84cc1615' },
  synthesize:     { icon: CheckCircle2,  color: '#22c55e', label: 'Synthesize',    bg: '#22c55e15' },
  default:        { icon: Cpu,           color: '#6b7280', label: 'Tool Call',     bg: '#6b728015' },
}

function getMeta(toolName) {
  if (!toolName) return TOOLS.default
  const key = Object.keys(TOOLS).find(k => toolName.toLowerCase().replace('_','').includes(k.replace('_','')))
           || Object.keys(TOOLS).find(k => k === toolName)
  return TOOLS[key] || TOOLS.default
}

// ── Parse "tool: input → output" strings ─────────────────────
function parseStep(str) {
  if (!str) return { tool: 'default', input: '', output: '', planKV: null }
  // Fix garbled arrow characters (encoding artifacts)
  const s = str.replace(/â/g, '→')
  const arrowIdx = s.indexOf(' → ')
  const left   = arrowIdx > -1 ? s.slice(0, arrowIdx).trim() : s.trim()
  const output = arrowIdx > -1 ? s.slice(arrowIdx + 3).trim() : ''
  const colonIdx = left.indexOf(': ')
  const tool   = colonIdx > -1 ? left.slice(0, colonIdx).trim() : 'default'
  const input  = colonIdx > -1 ? left.slice(colonIdx + 2).trim() : left

  // For llm_plan, try to parse plan dict into key-value pairs
  let planKV = null
  if (tool === 'llm_plan' && output.includes('{')) {
    planKV = {}
    const kv = [...output.matchAll(/["'\s]?(\w+)["'\s]?\s*:\s*["']?([^,'\s}]+)["']?/g)]
    kv.forEach(m => { if (m[1] && m[2] && m[2] !== '{') planKV[m[1]] = m[2] })
    if (Object.keys(planKV).length === 0) planKV = null
  }

  return { tool, input, output, planKV }
}

// ── Placeholder steps while running ──────────────────────────
const PLACEHOLDERS = [
  { tool: 'llm_plan',      input: 'Analysing question & building execution plan…', output: '', planKV: null },
  { tool: 'graph_filter',  input: 'Querying TigerGraph for matching events…',      output: '', planKV: null },
  { tool: 'sort_results',  input: 'Ranking by competitor count…',                  output: '', planKV: null },
  { tool: 'infobox_lookup',input: 'Fetching event infobox data…',                  output: '', planKV: null },
  { tool: 'synthesize',    input: 'Generating final answer…',                      output: '', planKV: null },
]

// ── Connector arrow between cards ─────────────────────────────
function Connector({ color }) {
  return (
    <div className="flex justify-center items-center h-7">
      <div className="flex flex-col items-center gap-0.5">
        <div className="w-px h-2" style={{ background: color + '50' }} />
        <ChevronDown size={12} style={{ color: color + '80' }} />
      </div>
    </div>
  )
}

// ── Single execution card ─────────────────────────────────────
function ExecCard({ step, state, index, totalSteps }) {
  const meta  = getMeta(step.tool)
  const Icon  = meta.icon
  const isActive  = state === 'active'
  const isDone    = state === 'done'
  const isPending = state === 'pending'

  return (
    <div
      className="rounded-xl border overflow-hidden transition-all duration-500"
      style={{
        borderColor:    isActive ? meta.color + '60' : isDone ? meta.color + '35' : '#1e3a5f',
        background:     isDone ? meta.bg : '#0a1525',
        opacity:        isPending ? 0.3 : 1,
        transform:      isPending ? 'translateY(6px)' : 'translateY(0)',
        boxShadow:      isActive ? `0 0 20px ${meta.color}25` : 'none',
      }}
    >
      {/* ── Card header ── */}
      <div
        className="flex items-center gap-3 px-4 py-3 border-b"
        style={{ borderColor: isDone ? meta.color + '25' : '#1e3a5f', background: meta.bg }}
      >
        {/* Step number */}
        <div className="w-5 h-5 rounded-full flex items-center justify-center shrink-0 text-[9px] font-bold"
             style={{ background: meta.color + '25', color: meta.color }}>
          {index + 1}
        </div>

        {/* Tool icon */}
        <div className="w-8 h-8 rounded-lg flex items-center justify-center shrink-0"
             style={{ background: meta.color + '20', border: `1px solid ${meta.color}30` }}>
          {isActive
            ? <Loader2 size={15} style={{ color: meta.color }} className="animate-spin" />
            : <Icon    size={15} style={{ color: isDone ? meta.color : '#4b5563'        }} />
          }
        </div>

        {/* Name */}
        <div className="flex-1 min-w-0">
          <p className="text-sm font-bold" style={{ color: isDone || isActive ? meta.color : '#6b7280' }}>
            {meta.label}
          </p>
          <p className="text-[10px] text-gray-600">step {index + 1} / {totalSteps}</p>
        </div>

        {/* Status pill */}
        <div className="shrink-0 px-2 py-0.5 rounded-full text-[9px] font-bold uppercase tracking-wider"
             style={{
               background: isActive ? meta.color + '20' : isDone ? meta.color + '15' : '#1e3a5f',
               color:      isActive ? meta.color        : isDone ? meta.color        : '#4b5563',
               border:     `1px solid ${isActive || isDone ? meta.color + '40' : '#1e3a5f'}`,
             }}>
          {isActive ? 'running' : isDone ? 'done' : 'queued'}
        </div>
      </div>

      {/* ── Input section ── */}
      <div className="px-4 pt-3 pb-2">
        <p className="text-[9px] text-gray-600 uppercase tracking-widest mb-1.5">Input</p>

        {/* LLM Plan: show parsed key-value pairs if available */}
        {step.planKV && isDone ? (
          <div className="grid grid-cols-2 gap-x-4 gap-y-1">
            {Object.entries(step.planKV).slice(0, 8).map(([k, v]) => (
              <div key={k} className="flex items-center gap-2">
                <span className="text-[10px] text-gray-600 shrink-0 w-14">{k}</span>
                <span className="text-[11px] font-semibold text-gray-200 font-mono">{String(v)}</span>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-[12px] text-gray-300 font-mono leading-relaxed"
             style={{ wordBreak: 'break-all' }}>
            {step.input || '…'}
          </p>
        )}
      </div>

      {/* ── Output section (only when done) ── */}
      {isDone && step.output && (
        <div className="mx-4 mb-3">
          <div className="h-px mb-2" style={{ background: meta.color + '30' }} />
          <p className="text-[9px] uppercase tracking-widest mb-1.5" style={{ color: meta.color + 'aa' }}>
            Output
          </p>
          <p className="text-[13px] font-semibold leading-snug" style={{ color: meta.color }}>
            {step.output}
          </p>
        </div>
      )}

      {/* Active: animated progress bar */}
      {isActive && (
        <div className="h-0.5 w-full overflow-hidden">
          <div className="h-full animate-pulse"
               style={{ background: `linear-gradient(90deg, transparent, ${meta.color}, transparent)`,
                        animation: 'sweep 1.6s ease-in-out infinite' }} />
        </div>
      )}
    </div>
  )
}

// ── Main component ────────────────────────────────────────────
export default function AgenticViz({ agentSteps = [], running, done }) {
  const [revealed,  setRevealed]  = useState(0)
  const [activeIdx, setActiveIdx] = useState(-1)
  const timers = useRef([])

  const steps   = done && agentSteps.length > 0 ? agentSteps.map(parseStep) : PLACEHOLDERS
  const isReal  = done && agentSteps.length > 0
  const nSteps  = steps.length

  useEffect(() => {
    timers.current.forEach(clearTimeout)
    timers.current = []
    setRevealed(0)
    setActiveIdx(-1)
    if (!done || !agentSteps.length) return

    const parsed = agentSteps.map(parseStep)
    parsed.forEach((_, i) => {
      const t1 = setTimeout(() => setActiveIdx(i),    i * 500)
      const t2 = setTimeout(() => setRevealed(i + 1), i * 500 + 440)
      timers.current.push(t1, t2)
    })
    const tf = setTimeout(() => setActiveIdx(-1), parsed.length * 500 + 440)
    timers.current.push(tf)
    return () => timers.current.forEach(clearTimeout)
  }, [done, JSON.stringify(agentSteps)])

  const getState = (i) => {
    if (!isReal) return running ? (i === 0 ? 'active' : 'pending') : 'pending'
    if (i < revealed)    return 'done'
    if (i === activeIdx) return 'active'
    return 'pending'
  }

  return (
    <div className="rounded-xl border overflow-hidden"
         style={{ background: '#060e1a', borderColor: '#1e3a5f' }}>

      {/* ── Terminal title bar ── */}
      <div className="flex items-center gap-3 px-4 py-2.5 border-b"
           style={{ borderColor: '#1e3a5f', background: '#08111e' }}>
        <div className="flex gap-1.5">
          {['#ff5f57','#febc2e','#28c840'].map((c, i) => (
            <div key={i} className="w-2.5 h-2.5 rounded-full" style={{ background: c, opacity: 0.75 }} />
          ))}
        </div>
        <span className="flex-1 text-center text-[10px] font-mono text-gray-500">
          agentic-graphrag — live execution
        </span>
        <div className="flex items-center gap-1.5">
          <div className="w-1.5 h-1.5 rounded-full"
               style={{
                 background: running && !done ? '#facc15' : done ? '#4ade80' : '#374151',
                 boxShadow:  running && !done ? '0 0 6px #facc15' : done ? '0 0 6px #4ade80' : 'none',
               }} />
          <span className="text-[10px] font-mono"
                style={{ color: running && !done ? '#facc15' : done ? '#4ade80' : '#4b5563' }}>
            {running && !done ? 'thinking…' : done ? `${agentSteps.length} steps · done` : 'idle'}
          </span>
        </div>
      </div>

      {/* ── Body ── */}
      <div className="p-4 space-y-0 overflow-y-auto" style={{ maxHeight: 540 }}>

        {/* Idle */}
        {!running && !done && (
          <div className="flex flex-col items-center justify-center py-16 gap-4">
            <div className="flex gap-3 opacity-20">
              {[TOOLS.llm_plan, TOOLS.graph_filter, TOOLS.sort_results].map((m, i) => {
                const I = m.icon
                return (
                  <div key={i} className="w-10 h-10 rounded-xl flex items-center justify-center border"
                       style={{ background: m.bg, borderColor: m.color + '30' }}>
                    <I size={18} style={{ color: m.color }} />
                  </div>
                )
              })}
            </div>
            <p className="text-xs text-gray-600 font-mono">ask a question to watch the agent plan and execute</p>
          </div>
        )}

        {/* Running or Done: show cards */}
        {(running || done) && steps.map((step, i) => (
          <div key={i}>
            <ExecCard
              step={step}
              state={getState(i)}
              index={i}
              totalSteps={nSteps}
            />
            {i < nSteps - 1 && (
              <Connector color={getMeta(step.tool).color} />
            )}
          </div>
        ))}
      </div>

      <style>{`
        @keyframes sweep {
          0%   { transform: translateX(-100%); }
          100% { transform: translateX(100%);  }
        }
      `}</style>
    </div>
  )
}
