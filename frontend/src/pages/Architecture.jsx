import { useResults } from '../hooks/useResults'

export default function Architecture() {
  const { data } = useResults()

  const grEM = data ? Math.round((data.graphrag?.overall?.exact_match ?? 0) * 100) : null
  const agEM = data ? Math.round((data.agentic?.overall?.exact_match  ?? 0) * 100) : null
  const rgEM = data ? Math.round((data.rag?.overall?.exact_match      ?? 0) * 100) : null

  const fmt = v => v == null ? '—%' : `${v}%`

  const pipelines = [
    {
      name: 'RAG (Baseline)',
      color: '#f97316',
      bg: 'rgba(249,115,22,0.08)',
      border: 'rgba(249,115,22,0.3)',
      accuracy: fmt(rgEM),
      tokens: '~1,856/q',
      steps: [
        { label: 'Question', color: '#374151' },
        { label: 'nomic-embed-text\n768-dim embedding', color: '#1d4ed8' },
        { label: 'Cosine similarity\ntop-5 docs', color: '#1d4ed8' },
        { label: 'qwen3:4b LLM\nanswer synthesis', color: '#7c3aed' },
        { label: 'Answer', color: '#374151' },
      ],
      weakness: 'Cannot count across full corpus. Top-5 docs miss events for aggregation/superlative. No structured reasoning.',
    },
    {
      name: 'GraphRAG',
      color: '#0091ff',
      bg: 'rgba(0,145,255,0.08)',
      border: 'rgba(0,145,255,0.3)',
      accuracy: fmt(grEM),
      tokens: '0/q',
      steps: [
        { label: 'Question', color: '#374151' },
        { label: 'Regex entity\nextraction', color: '#065f46' },
        { label: 'TigerGraph\nREST++ query', color: '#0074cc' },
        { label: 'Edge traversal\nPREV/NEXT_EDITION', color: '#0074cc' },
        { label: 'Infobox parsing\ndirect answer', color: '#065f46' },
      ],
      weakness: 'Struggles with ambiguous multi-hop questions where multiple events share the same venue and date.',
    },
    {
      name: 'Agentic GraphRAG',
      color: '#a855f7',
      bg: 'rgba(168,85,247,0.08)',
      border: 'rgba(168,85,247,0.3)',
      accuracy: fmt(agEM),
      tokens: 'LLM calls/q',
      steps: [
        { label: 'Question', color: '#374151' },
        { label: 'LLM plan\n(JSON intent)', color: '#7c3aed' },
        { label: 'Tool dispatch:\ngraph + vector', color: '#0074cc' },
        { label: 'Evidence chain\nsynthesis', color: '#7c3aed' },
        { label: 'Answer', color: '#374151' },
      ],
      weakness: 'Inherits multi-hop limitation from graph; LLM adds latency and tokens.',
    },
  ]

  const graphStats = [
    { label: 'OlympicEvent vertices', value: '2,187' },
    { label: 'Medalist vertices',     value: '5,359' },
    { label: 'Venue vertices',        value: '319'   },
    { label: 'Corpus documents',      value: '2,951' },
  ]

  return (
    <div className="p-6 space-y-8 max-w-5xl mx-auto">
      <div>
        <h1 className="text-2xl font-bold text-white">System Architecture</h1>
        <p className="text-gray-400 text-sm mt-1">Three retrieval strategies over the same Olympic knowledge base</p>
      </div>

      {/* Graph schema */}
      <div className="bg-surface-800 border border-surface-600 rounded-xl p-6">
        <h2 className="text-sm font-semibold text-gray-300 mb-5">TigerGraph Schema</h2>
        <div className="grid grid-cols-2 gap-6">
          {/* Vertex / Edge diagram */}
          <div className="space-y-3">
            <div className="bg-brand-500/10 border border-brand-500/30 rounded-lg p-4">
              <p className="text-xs font-semibold text-brand-400 mb-2">OlympicEvent (vertex)</p>
              <div className="grid grid-cols-2 gap-1">
                {['title', 'year', 'season', 'sport', 'event', 'venue_name', 'competitors', 'nations'].map(f => (
                  <span key={f} className="text-xs font-mono text-gray-400 bg-surface-700 px-2 py-0.5 rounded">{f}</span>
                ))}
              </div>
            </div>
            <div className="flex gap-3">
              <div className="flex-1 bg-purple-500/10 border border-purple-500/30 rounded-lg p-3">
                <p className="text-xs font-semibold text-purple-400 mb-1.5">Medalist (vertex)</p>
                <p className="text-xs font-mono text-gray-500">name · country</p>
              </div>
              <div className="flex-1 bg-green-500/10 border border-green-500/30 rounded-lg p-3">
                <p className="text-xs font-semibold text-green-400 mb-1.5">Venue (vertex)</p>
                <p className="text-xs font-mono text-gray-500">name · city</p>
              </div>
            </div>
            <div className="bg-yellow-500/10 border border-yellow-500/30 rounded-lg p-4">
              <p className="text-xs font-semibold text-yellow-400 mb-2">Edges</p>
              <div className="space-y-1.5 text-xs font-mono text-gray-400">
                <p>OlympicEvent → <span className="text-yellow-300">HELD_AT</span> → Venue</p>
                <p>OlympicEvent → <span className="text-cyan-300">PREV_EDITION</span> → OlympicEvent</p>
                <p>OlympicEvent → <span className="text-cyan-300">NEXT_EDITION</span> → OlympicEvent</p>
              </div>
            </div>
          </div>
          {/* Stats */}
          <div className="space-y-3">
            <p className="text-xs text-gray-500 uppercase tracking-wider mb-3">Graph Statistics</p>
            {graphStats.map(({ label, value }) => (
              <div key={label} className="flex items-center justify-between bg-surface-700 rounded-lg px-4 py-3">
                <span className="text-sm text-gray-400">{label}</span>
                <span className="text-lg font-bold text-white">{value}</span>
              </div>
            ))}
            <div className="bg-cyan-500/10 border border-cyan-500/30 rounded-lg p-3 mt-2">
              <p className="text-xs text-cyan-400 font-semibold mb-1">Key Innovation</p>
              <p className="text-xs text-gray-400 leading-relaxed">
                PREV/NEXT_EDITION edges encode the Olympic calendar — enabling perfect temporal chain traversal without any LLM reasoning.
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Pipeline diagrams */}
      <div className="space-y-4">
        <h2 className="text-sm font-semibold text-gray-300">Pipeline Comparison</h2>
        {pipelines.map(p => (
          <div key={p.name} style={{ background: p.bg, borderColor: p.border }}
               className="border rounded-xl p-5">
            <div className="flex items-center justify-between mb-5">
              <div className="flex items-center gap-3">
                <div className="w-2 h-8 rounded-full" style={{ background: p.color }} />
                <div>
                  <h3 className="font-semibold text-white">{p.name}</h3>
                  <p className="text-xs text-gray-500">{p.weakness}</p>
                </div>
              </div>
              <div className="flex gap-4 text-right">
                <div>
                  <p className="text-xs text-gray-500">Accuracy</p>
                  <p className="text-xl font-bold" style={{ color: p.color }}>{p.accuracy}</p>
                </div>
                <div>
                  <p className="text-xs text-gray-500">LLM tokens</p>
                  <p className="text-xl font-bold text-white">{p.tokens}</p>
                </div>
              </div>
            </div>
            {/* Flow steps */}
            <div className="flex items-center gap-2 overflow-x-auto pb-1">
              {p.steps.map((step, i) => (
                <div key={i} className="flex items-center gap-2 shrink-0">
                  <div className="text-center">
                    <div
                      className="px-4 py-2.5 rounded-lg text-xs font-medium text-white whitespace-pre-line text-center min-w-[100px]"
                      style={{ background: step.color, opacity: 0.9 }}
                    >
                      {step.label}
                    </div>
                  </div>
                  {i < p.steps.length - 1 && (
                    <svg width="20" height="12" viewBox="0 0 20 12" fill="none">
                      <path d="M0 6H16M16 6L11 1M16 6L11 11" stroke="#4b5563" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                    </svg>
                  )}
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>

      {/* Models */}
      <div className="bg-surface-800 border border-surface-600 rounded-xl p-5">
        <h2 className="text-sm font-semibold text-gray-300 mb-4">Models & Infrastructure</h2>
        <div className="grid grid-cols-3 gap-4">
          {[
            { name: 'nomic-embed-text', role: 'Document + query embeddings', detail: '768-dim · 137M params · Ollama', color: 'green' },
            { name: 'qwen3:4b',         role: 'LLM for reasoning & planning', detail: '4B params · Ollama (local)',      color: 'purple'},
            { name: 'TigerGraph Savanna', role: 'Graph database',            detail: 'v4.2.5 · REST++ · JWT auth',       color: 'brand' },
          ].map(m => (
            <div key={m.name} className="bg-surface-700 rounded-lg p-4">
              <p className={`text-xs font-semibold mb-1 text-${m.color}-400`}>{m.name}</p>
              <p className="text-sm text-gray-300 mb-1">{m.role}</p>
              <p className="text-xs text-gray-500">{m.detail}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
