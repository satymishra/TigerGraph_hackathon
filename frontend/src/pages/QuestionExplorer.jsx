import { useState, useMemo } from 'react'
import { useResults } from '../hooks/useResults'
import { ChevronRight, CheckCircle, XCircle, Search, X } from 'lucide-react'

const TYPE_COLORS = {
  aggregation: 'bg-blue-500/15 text-blue-400 border-blue-500/30',
  lookup:      'bg-green-500/15 text-green-400 border-green-500/30',
  multi_hop:   'bg-purple-500/15 text-purple-400 border-purple-500/30',
  superlative: 'bg-yellow-500/15 text-yellow-400 border-yellow-500/30',
  temporal:    'bg-cyan-500/15 text-cyan-400 border-cyan-500/30',
}

const PIPELINE_META = [
  { key: 'graphrag', label: 'GraphRAG',  color: 'text-brand-400'  },
  { key: 'agentic',  label: 'Agentic',   color: 'text-purple-400' },
  { key: 'rag',      label: 'RAG',       color: 'text-orange-400' },
]

function Badge({ type }) {
  return (
    <span className={`text-[10px] font-semibold px-2 py-0.5 rounded-full border ${TYPE_COLORS[type] || 'bg-gray-500/15 text-gray-400 border-gray-500/30'}`}>
      {type}
    </span>
  )
}

function ResultDot({ em }) {
  return em
    ? <CheckCircle size={14} className="text-green-400 shrink-0" />
    : <XCircle    size={14} className="text-red-400  shrink-0" />
}

function DetailPanel({ qid, byPipeline, onClose }) {
  const { gr, ag, rg } = byPipeline

  const pipelineResult = (res, label, color) => {
    if (!res) return null
    const graphPath = res.graph_path || res.evidence_chain || ''
    const steps = graphPath.split('→').map(s => s.trim()).filter(Boolean)
    return (
      <div className="bg-surface-700/50 border border-surface-600 rounded-lg p-4">
        <div className="flex items-center justify-between mb-3">
          <span className={`text-sm font-semibold ${color}`}>{label}</span>
          <ResultDot em={res.exact_match} />
        </div>
        <div className="mb-3">
          <span className="text-xs text-gray-500 uppercase tracking-wider">Answer</span>
          <p className={`text-sm font-medium mt-1 ${res.exact_match ? 'text-green-300' : 'text-red-300'}`}>
            {res.predicted_answer || '—'}
          </p>
          <p className="text-xs text-gray-500 mt-0.5">Gold: {res.gold_answer?.join(', ')}</p>
        </div>
        {steps.length > 0 && (
          <div>
            <span className="text-xs text-gray-500 uppercase tracking-wider">Graph Path</span>
            <div className="mt-2 space-y-1.5">
              {steps.map((step, i) => (
                <div key={i} className="flex items-start gap-2">
                  <div className="mt-1.5 w-1.5 h-1.5 rounded-full bg-brand-500/60 shrink-0" />
                  <p className="text-xs text-gray-400 font-mono leading-relaxed">{step}</p>
                </div>
              ))}
            </div>
          </div>
        )}
        {res.tokens_used > 0 && (
          <p className="text-xs text-gray-600 mt-3">Tokens: {res.tokens_used} · {res.elapsed_sec}s</p>
        )}
      </div>
    )
  }

  return (
    <div className="w-[420px] shrink-0 bg-surface-800 border-l border-surface-600 flex flex-col overflow-hidden">
      <div className="flex items-center justify-between px-5 py-4 border-b border-surface-600">
        <span className="text-sm font-semibold text-gray-200">{qid}</span>
        <button onClick={onClose} className="text-gray-500 hover:text-gray-300">
          <X size={16} />
        </button>
      </div>
      <div className="flex-1 overflow-y-auto p-4 space-y-3">
        <div className="bg-surface-700 rounded-lg p-3 mb-4">
          <p className="text-xs text-gray-500 uppercase tracking-wider mb-1.5">Question</p>
          <p className="text-sm text-gray-200 leading-relaxed">{gr?.question || ag?.question || rg?.question}</p>
          <div className="mt-2">
            <Badge type={gr?.qtype || ag?.qtype || rg?.qtype} />
          </div>
        </div>
        {pipelineResult(gr, 'GraphRAG', 'text-brand-400')}
        {pipelineResult(ag, 'Agentic GraphRAG', 'text-purple-400')}
        {pipelineResult(rg, 'RAG (baseline)', 'text-orange-400')}
      </div>
    </div>
  )
}

export default function QuestionExplorer() {
  const { data, loading, error } = useResults()
  const [search, setSearch]     = useState('')
  const [typeFilter, setTypeFilter] = useState('all')
  const [statusFilter, setStatusFilter] = useState('all')
  const [selected, setSelected] = useState(null)

  const questionMap = useMemo(() => {
    if (!data) return {}
    const map = {}
    for (const [key, pData] of Object.entries(data)) {
      for (const r of (pData?.results || [])) {
        if (!map[r.qid]) map[r.qid] = {}
        map[r.qid][key] = r
      }
    }
    return map
  }, [data])

  const questions = useMemo(() => {
    return Object.entries(questionMap).map(([qid, pipelines]) => ({
      qid,
      question: pipelines.graphrag?.question || pipelines.agentic?.question || pipelines.rag?.question || '',
      qtype:    pipelines.graphrag?.qtype    || pipelines.agentic?.qtype    || pipelines.rag?.qtype    || '',
      grEM: pipelines.graphrag?.exact_match ?? null,
      agEM: pipelines.agentic?.exact_match  ?? null,
      rgEM: pipelines.rag?.exact_match      ?? null,
      gold: pipelines.graphrag?.gold_answer || [],
      pipelines,
    })).sort((a, b) => a.qid.localeCompare(b.qid))
  }, [questionMap])

  const filtered = useMemo(() => {
    return questions.filter(q => {
      if (search && !q.question.toLowerCase().includes(search.toLowerCase())) return false
      if (typeFilter !== 'all' && q.qtype !== typeFilter) return false
      if (statusFilter === 'correct'   && q.grEM !== 1) return false
      if (statusFilter === 'incorrect' && q.grEM !== 0) return false
      return true
    })
  }, [questions, search, typeFilter, statusFilter])

  const types = ['all', 'aggregation', 'lookup', 'multi_hop', 'superlative', 'temporal']

  if (loading) return (
    <div className="flex items-center justify-center h-full">
      <div className="w-6 h-6 border-2 border-brand-500 border-t-transparent rounded-full animate-spin" />
    </div>
  )
  if (error) return <div className="p-6 text-red-400 text-sm">Error: {error}</div>

  const selectedData = selected ? {
    gr: questionMap[selected]?.graphrag,
    ag: questionMap[selected]?.agentic,
    rg: questionMap[selected]?.rag,
  } : null

  return (
    <div className="flex h-full overflow-hidden">
      {/* List */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Filters */}
        <div className="px-5 py-4 border-b border-surface-600 bg-surface-800/50 space-y-3">
          <div className="flex items-center justify-between">
            <h1 className="text-lg font-bold text-white">Question Explorer</h1>
            <span className="text-sm text-gray-500">{filtered.length} / {questions.length} questions</span>
          </div>
          <div className="flex gap-3">
            <div className="flex-1 relative">
              <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500" />
              <input
                value={search}
                onChange={e => setSearch(e.target.value)}
                placeholder="Search questions…"
                className="w-full pl-8 pr-3 py-2 bg-surface-700 border border-surface-600 rounded-lg text-sm text-gray-200 placeholder-gray-500 focus:outline-none focus:border-brand-500"
              />
            </div>
            <select
              value={statusFilter}
              onChange={e => setStatusFilter(e.target.value)}
              className="bg-surface-700 border border-surface-600 rounded-lg px-3 py-2 text-sm text-gray-300 focus:outline-none"
            >
              <option value="all">All status</option>
              <option value="correct">GraphRAG ✓</option>
              <option value="incorrect">GraphRAG ✗</option>
            </select>
          </div>
          <div className="flex gap-1.5 flex-wrap">
            {types.map(t => (
              <button
                key={t}
                onClick={() => setTypeFilter(t)}
                className={`text-xs px-3 py-1.5 rounded-full border transition-colors ${
                  typeFilter === t
                    ? 'bg-brand-500/20 text-brand-400 border-brand-500/40'
                    : 'bg-surface-700 text-gray-400 border-surface-600 hover:border-surface-500'
                }`}
              >
                {t === 'all' ? 'All types' : t}
              </button>
            ))}
          </div>
        </div>

        {/* Table */}
        <div className="flex-1 overflow-y-auto">
          <table className="w-full text-sm">
            <thead className="sticky top-0 bg-surface-800 z-10">
              <tr className="border-b border-surface-600">
                <th className="text-left px-5 py-3 text-xs font-semibold text-gray-500 uppercase tracking-wider w-20">QID</th>
                <th className="text-left px-3 py-3 text-xs font-semibold text-gray-500 uppercase tracking-wider">Question</th>
                <th className="text-center px-3 py-3 text-xs font-semibold text-gray-500 uppercase tracking-wider w-24">Type</th>
                <th className="text-center px-3 py-3 text-xs font-semibold text-[#0091ff] uppercase tracking-wider w-16">GR</th>
                <th className="text-center px-3 py-3 text-xs font-semibold text-purple-400 uppercase tracking-wider w-16">AG</th>
                <th className="text-center px-3 py-3 text-xs font-semibold text-orange-400 uppercase tracking-wider w-16">RAG</th>
                <th className="w-8" />
              </tr>
            </thead>
            <tbody>
              {filtered.map((q, i) => (
                <tr
                  key={q.qid}
                  onClick={() => setSelected(q.qid === selected ? null : q.qid)}
                  className={`border-b border-surface-700 cursor-pointer transition-colors ${
                    q.qid === selected
                      ? 'bg-brand-500/10'
                      : i % 2 === 0 ? 'hover:bg-surface-700/50' : 'bg-surface-700/20 hover:bg-surface-700/60'
                  }`}
                >
                  <td className="px-5 py-3 font-mono text-xs text-gray-500">{q.qid}</td>
                  <td className="px-3 py-3 text-gray-300 max-w-xs">
                    <p className="truncate">{q.question}</p>
                    <p className="text-xs text-gray-600 truncate mt-0.5">Gold: {q.gold.join(', ')}</p>
                  </td>
                  <td className="px-3 py-3 text-center"><Badge type={q.qtype} /></td>
                  <td className="px-3 py-3 text-center"><ResultDot em={q.grEM} /></td>
                  <td className="px-3 py-3 text-center"><ResultDot em={q.agEM} /></td>
                  <td className="px-3 py-3 text-center"><ResultDot em={q.rgEM} /></td>
                  <td className="px-2 py-3"><ChevronRight size={14} className="text-gray-600" /></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Detail panel */}
      {selected && selectedData && (
        <DetailPanel
          qid={selected}
          byPipeline={selectedData}
          onClose={() => setSelected(null)}
        />
      )}
    </div>
  )
}
