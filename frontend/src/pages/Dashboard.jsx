import { useResults } from '../hooks/useResults'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  RadarChart, Radar, PolarGrid, PolarAngleAxis, ResponsiveContainer,
} from 'recharts'
import { TrendingUp, Zap, Layers, Clock } from 'lucide-react'

const PIPELINE_COLORS = {
  GraphRAG: '#0091ff',
  Agentic:  '#a855f7',
  RAG:      '#f97316',
}

const TYPE_LABELS = {
  aggregation: 'Aggregation',
  lookup:      'Lookup',
  multi_hop:   'Multi-hop',
  superlative: 'Superlative',
  temporal:    'Temporal',
}

function StatCard({ icon: Icon, label, value, sub, color = 'brand' }) {
  const colors = {
    brand:  'from-brand-500/20 to-brand-500/5 border-brand-500/30 text-brand-400',
    purple: 'from-purple-500/20 to-purple-500/5 border-purple-500/30 text-purple-400',
    orange: 'from-orange-500/20 to-orange-500/5 border-orange-500/30 text-orange-400',
    green:  'from-green-500/20 to-green-500/5 border-green-500/30 text-green-400',
  }
  return (
    <div className={`bg-gradient-to-br ${colors[color]} border rounded-xl p-5`}>
      <div className="flex items-start justify-between">
        <div>
          <p className="text-xs text-gray-400 font-medium uppercase tracking-wider mb-1">{label}</p>
          <p className="text-3xl font-bold text-white">{value}</p>
          {sub && <p className="text-xs text-gray-500 mt-1">{sub}</p>}
        </div>
        <div className={`p-2.5 rounded-lg bg-black/20`}>
          <Icon size={18} className={colors[color].split(' ').pop()} />
        </div>
      </div>
    </div>
  )
}

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null
  return (
    <div className="bg-surface-700 border border-surface-600 rounded-lg p-3 text-sm shadow-xl">
      <p className="text-gray-300 font-medium mb-2">{TYPE_LABELS[label] || label}</p>
      {payload.map(p => (
        <div key={p.dataKey} className="flex items-center gap-2 mb-1">
          <div className="w-2.5 h-2.5 rounded-sm" style={{ background: p.fill }} />
          <span className="text-gray-400">{p.name}:</span>
          <span className="text-white font-medium">{(p.value * 100).toFixed(1)}%</span>
        </div>
      ))}
    </div>
  )
}

export default function Dashboard() {
  const { data, loading, error } = useResults()

  if (loading) return (
    <div className="flex items-center justify-center h-full">
      <div className="text-center">
        <div className="w-8 h-8 border-2 border-brand-500 border-t-transparent rounded-full animate-spin mx-auto mb-3" />
        <p className="text-gray-400 text-sm">Loading benchmark results…</p>
      </div>
    </div>
  )

  if (error) return (
    <div className="flex items-center justify-center h-full">
      <div className="text-center text-red-400 text-sm">
        <p className="font-medium mb-1">Backend not reachable</p>
        <p className="text-gray-500">Run: <code className="bg-surface-700 px-2 py-0.5 rounded">uvicorn backend.main:app --port 8000</code></p>
      </div>
    </div>
  )

  const gr = data.graphrag
  const ag = data.agentic
  const rg = data.rag

  // Per-type bar chart data
  const types = ['aggregation', 'lookup', 'multi_hop', 'superlative', 'temporal']
  const typeData = types.map(t => ({
    type: t,
    GraphRAG: gr.by_type?.[t]?.exact_match ?? 0,
    Agentic:  ag.by_type?.[t]?.exact_match ?? 0,
    RAG:      rg.by_type?.[t]?.exact_match ?? 0,
  }))

  // Radar data
  const radarData = types.map(t => ({
    subject: TYPE_LABELS[t],
    GraphRAG: Math.round((gr.by_type?.[t]?.exact_match ?? 0) * 100),
    Agentic:  Math.round((ag.by_type?.[t]?.exact_match ?? 0) * 100),
    RAG:      Math.round((rg.by_type?.[t]?.exact_match ?? 0) * 100),
  }))

  const grEM = Math.round((gr.overall?.exact_match ?? 0) * 100)
  const agEM = Math.round((ag.overall?.exact_match ?? 0) * 100)
  const rgEM = Math.round((rg.overall?.exact_match ?? 0) * 100)

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-white">Benchmark Results</h1>
        <p className="text-gray-400 text-sm mt-1">
          3 pipelines · 100 Olympic questions · 5 question types
        </p>
      </div>

      {/* Stat cards */}
      <div className="grid grid-cols-4 gap-4">
        <StatCard icon={TrendingUp} label="GraphRAG Accuracy"  value={`${grEM}%`} sub="Exact match · 0 LLM tokens" color="brand"  />
        <StatCard icon={Layers}    label="Agentic GraphRAG"    value={`${agEM}%`} sub="Adaptive multi-step reasoning" color="purple"/>
        <StatCard icon={Clock}     label="RAG Baseline"        value={`${rgEM}%`} sub="Vector search · ~1,856 tokens/q" color="orange"/>
        <StatCard icon={Zap}       label="GraphRAG vs RAG"     value={`+${grEM - rgEM}pp`} sub="Accuracy gain over RAG baseline" color="green" />
      </div>

      {/* Overall comparison bar */}
      <div className="bg-surface-800 border border-surface-600 rounded-xl p-5">
        <h2 className="text-sm font-semibold text-gray-300 mb-4">Overall Exact Match</h2>
        <div className="space-y-3">
          {[
            { name: 'GraphRAG', em: grEM, color: '#0091ff' },
            { name: 'Agentic GraphRAG', em: agEM, color: '#a855f7' },
            { name: 'RAG (baseline)', em: rgEM, color: '#f97316' },
          ].map(({ name, em, color }) => (
            <div key={name} className="flex items-center gap-4">
              <span className="text-sm text-gray-400 w-36 shrink-0">{name}</span>
              <div className="flex-1 bg-surface-700 rounded-full h-6 overflow-hidden">
                <div
                  className="h-full rounded-full flex items-center justify-end pr-3 transition-all duration-700"
                  style={{ width: `${em}%`, background: color }}
                >
                  <span className="text-xs font-bold text-white">{em}%</span>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Charts row */}
      <div className="grid grid-cols-2 gap-4">
        {/* Grouped bar chart */}
        <div className="bg-surface-800 border border-surface-600 rounded-xl p-5">
          <h2 className="text-sm font-semibold text-gray-300 mb-4">Accuracy by Question Type</h2>
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={typeData} barCategoryGap="25%" barGap={3}>
              <CartesianGrid strokeDasharray="3 3" stroke="#21262d" />
              <XAxis
                dataKey="type"
                tick={{ fill: '#8b949e', fontSize: 11 }}
                tickFormatter={t => TYPE_LABELS[t]?.split('-')[0] ?? t}
                axisLine={{ stroke: '#30363d' }}
                tickLine={false}
              />
              <YAxis
                tickFormatter={v => `${Math.round(v * 100)}%`}
                tick={{ fill: '#8b949e', fontSize: 11 }}
                axisLine={false}
                tickLine={false}
                domain={[0, 1]}
              />
              <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(255,255,255,0.03)' }} />
              <Legend
                wrapperStyle={{ fontSize: '12px', color: '#8b949e', paddingTop: '8px' }}
              />
              {Object.entries(PIPELINE_COLORS).map(([name, color]) => (
                <Bar key={name} dataKey={name} fill={color} radius={[3, 3, 0, 0]} />
              ))}
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Radar chart */}
        <div className="bg-surface-800 border border-surface-600 rounded-xl p-5">
          <h2 className="text-sm font-semibold text-gray-300 mb-4">Capability Radar</h2>
          <ResponsiveContainer width="100%" height={240}>
            <RadarChart data={radarData}>
              <PolarGrid stroke="#30363d" />
              <PolarAngleAxis dataKey="subject" tick={{ fill: '#8b949e', fontSize: 11 }} />
              {Object.entries(PIPELINE_COLORS).map(([name, color]) => (
                <Radar
                  key={name}
                  name={name}
                  dataKey={name}
                  stroke={color}
                  fill={color}
                  fillOpacity={0.12}
                  strokeWidth={2}
                />
              ))}
              <Legend wrapperStyle={{ fontSize: '12px', color: '#8b949e' }} />
            </RadarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Detailed table */}
      <div className="bg-surface-800 border border-surface-600 rounded-xl overflow-hidden">
        <div className="px-5 py-4 border-b border-surface-600">
          <h2 className="text-sm font-semibold text-gray-300">Detailed Breakdown</h2>
        </div>
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-surface-600">
              <th className="text-left px-5 py-3 text-xs font-semibold text-gray-500 uppercase tracking-wider">Type</th>
              <th className="text-center px-4 py-3 text-xs font-semibold text-gray-500 uppercase tracking-wider">N</th>
              <th className="text-center px-4 py-3 text-xs font-semibold text-[#0091ff] uppercase tracking-wider">GraphRAG</th>
              <th className="text-center px-4 py-3 text-xs font-semibold text-purple-400 uppercase tracking-wider">Agentic</th>
              <th className="text-center px-4 py-3 text-xs font-semibold text-orange-400 uppercase tracking-wider">RAG</th>
              <th className="text-left px-5 py-3 text-xs font-semibold text-gray-500 uppercase tracking-wider">Key Insight</th>
            </tr>
          </thead>
          <tbody>
            {types.map((t, i) => {
              const grVal = Math.round((gr.by_type?.[t]?.exact_match ?? 0) * 100)
              const agVal = Math.round((ag.by_type?.[t]?.exact_match ?? 0) * 100)
              const rgVal = Math.round((rg.by_type?.[t]?.exact_match ?? 0) * 100)
              const n = gr.by_type?.[t]?.n ?? 0
              const insights = {
                aggregation: 'RAG fails — needs full corpus count, not top-5 docs',
                lookup:      'All methods find exact titles well',
                multi_hop:   'Graph venue+date traversal outperforms retrieval',
                superlative: 'Graph sort beats LLM reasoning over passages',
                temporal:    'PREV/NEXT_EDITION edges give perfect chain traversal',
              }
              const cell = (v) => (
                <span className={`font-semibold ${v >= 90 ? 'text-green-400' : v >= 60 ? 'text-yellow-400' : 'text-red-400'}`}>
                  {v}%
                </span>
              )
              return (
                <tr key={t} className={`border-b border-surface-700 ${i % 2 === 0 ? '' : 'bg-surface-700/30'}`}>
                  <td className="px-5 py-3 font-medium text-gray-200">{TYPE_LABELS[t]}</td>
                  <td className="px-4 py-3 text-center text-gray-500">{n}</td>
                  <td className="px-4 py-3 text-center">{cell(grVal)}</td>
                  <td className="px-4 py-3 text-center">{cell(agVal)}</td>
                  <td className="px-4 py-3 text-center">{cell(rgVal)}</td>
                  <td className="px-5 py-3 text-gray-500 text-xs">{insights[t]}</td>
                </tr>
              )
            })}
            <tr className="bg-surface-700/50">
              <td className="px-5 py-3 font-bold text-white">Overall</td>
              <td className="px-4 py-3 text-center text-gray-400">100</td>
              <td className="px-4 py-3 text-center font-bold text-brand-400">{grEM}%</td>
              <td className="px-4 py-3 text-center font-bold text-purple-400">{agEM}%</td>
              <td className="px-4 py-3 text-center font-bold text-orange-400">{rgEM}%</td>
              <td className="px-5 py-3 text-gray-500 text-xs">GraphRAG is {Math.round(grEM / Math.max(1, rgEM))}× more accurate than RAG</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  )
}
