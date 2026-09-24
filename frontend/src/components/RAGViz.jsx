import { useEffect, useState } from 'react'
import { FileText, Search, Layers } from 'lucide-react'

const ACCENT = '#f97316' // orange

function DocCard({ doc, rank, visible, animDone }) {
  // Rank-based fill: rank 1 = 100%, rank 5 = 30%
  const barFill = Math.max(30, 100 - (rank - 1) * 16)

  return (
    <div
      className="flex items-start gap-3 p-3 rounded-xl border transition-all duration-500"
      style={{
        borderColor: visible ? '#f97316' + '35' : 'transparent',
        background: visible ? '#f9731608' : 'transparent',
        opacity: visible ? 1 : 0,
        transform: visible ? 'translateX(0)' : 'translateX(-16px)',
        transitionDelay: `${(rank - 1) * 140}ms`,
      }}
    >
      {/* Rank badge */}
      <div
        className="flex-shrink-0 w-6 h-6 rounded-md flex items-center justify-center text-[10px] font-bold mt-0.5"
        style={{ background: '#f97316' + (rank === 1 ? '30' : '18'), color: ACCENT }}
      >
        {rank}
      </div>

      {/* Content */}
      <div className="flex-1 min-w-0">
        <p className="text-xs font-medium text-gray-300 leading-snug truncate">
          {doc.title || `Document ${rank}`}
        </p>

        {/* Relevance bar */}
        {animDone && (
          <div className="mt-2 flex items-center gap-2">
            <div className="flex-1 h-1 bg-surface-600 rounded-full overflow-hidden">
              <div
                className="h-full rounded-full"
                style={{
                  width: `${barFill}%`,
                  background: `linear-gradient(90deg, #f9731680, ${ACCENT})`,
                  transition: 'width 0.7s ease',
                  transitionDelay: `${(rank - 1) * 140 + 300}ms`,
                }}
              />
            </div>
            <span className="text-[10px] text-gray-500 font-mono w-8 text-right">
              #{rank}
            </span>
          </div>
        )}
      </div>
    </div>
  )
}

function SearchingState() {
  return (
    <div className="flex flex-col items-center justify-center py-10 gap-4">
      <div className="relative w-12 h-12">
        <div
          className="absolute inset-0 rounded-full border-2 animate-ping"
          style={{ borderColor: ACCENT + '40' }}
        />
        <div
          className="absolute inset-1 rounded-full border-2 animate-pulse"
          style={{ borderColor: ACCENT + '70' }}
        />
        <Search size={16} className="absolute inset-0 m-auto" style={{ color: ACCENT }} />
      </div>
      <div className="space-y-1 text-center">
        <p className="text-xs text-gray-400 animate-pulse">Searching vector space…</p>
        <p className="text-[10px] text-gray-600">nomic-embed-text · cosine similarity</p>
      </div>
      {/* Animated embedding dots */}
      <div className="flex gap-1.5 mt-1">
        {Array.from({ length: 12 }).map((_, i) => (
          <div
            key={i}
            className="w-1 rounded-full"
            style={{
              height: `${6 + Math.sin(i * 0.8) * 5}px`,
              background: ACCENT,
              opacity: 0.3 + Math.sin(i * 0.8) * 0.3,
              animation: `ragWave 1.2s ease-in-out infinite alternate`,
              animationDelay: `${i * 80}ms`,
            }}
          />
        ))}
      </div>
    </div>
  )
}

function IdleState() {
  return (
    <div className="flex flex-col items-center justify-center py-10 gap-3">
      <Layers size={28} className="text-gray-700" />
      <div className="text-center">
        <p className="text-xs text-gray-600">Retrieval-Augmented Generation</p>
        <p className="text-[10px] text-gray-700 mt-0.5">
          Embeds your query · searches {' '}
          <span className="text-gray-600">top-5 docs</span> · LLM answers
        </p>
      </div>
      {/* Ghost doc rows */}
      <div className="w-full space-y-1.5 mt-2 opacity-15">
        {Array.from({ length: 4 }).map((_, i) => (
          <div key={i} className="h-8 rounded-lg bg-surface-700" />
        ))}
      </div>
    </div>
  )
}

export default function RAGViz({ docSnippets, running, done }) {
  const [visibleCount, setVisibleCount] = useState(0)
  const [barsVisible, setBarsVisible] = useState(false)

  const docs = docSnippets?.length
    ? docSnippets
    : Array.from({ length: 5 }, (_, i) => ({ title: `Document ${i + 1}`, doc_id: `d${i}` }))

  useEffect(() => {
    if (!done) {
      setVisibleCount(0)
      setBarsVisible(false)
      return
    }
    // Stagger cards in
    let count = 0
    const iv = setInterval(() => {
      count++
      setVisibleCount(count)
      if (count >= docs.length) {
        clearInterval(iv)
        setTimeout(() => setBarsVisible(true), 300)
      }
    }, 160)
    return () => clearInterval(iv)
  }, [done, JSON.stringify(docSnippets)])

  const phase = !running && !done ? 'idle' : running && !done ? 'searching' : 'done'

  return (
    <div className="rounded-xl bg-[#080d14] border border-surface-700 overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-surface-700/60">
        <div className="flex items-center gap-2">
          <div
            className={`w-2 h-2 rounded-full ${
              phase === 'searching' ? 'animate-pulse' : ''
            }`}
            style={{
              background: phase === 'done' ? '#10b981' : phase === 'searching' ? ACCENT : '#374151',
            }}
          />
          <span className="text-xs font-semibold text-gray-400 uppercase tracking-wider">
            {phase === 'idle'      && 'Vector Retrieval'}
            {phase === 'searching' && 'Searching…'}
            {phase === 'done'      && `${docs.length} Documents Retrieved`}
          </span>
        </div>
        {phase === 'done' && (
          <span className="text-[10px] font-mono" style={{ color: ACCENT + 'cc' }}>
            nomic-embed-text
          </span>
        )}
      </div>

      {/* Body */}
      <div className="p-3">
        {phase === 'idle'      && <IdleState />}
        {phase === 'searching' && <SearchingState />}
        {phase === 'done' && (
          <div className="space-y-1">
            {docs.map((doc, i) => (
              <DocCard
                key={doc.doc_id || i}
                doc={doc}
                rank={i + 1}
                visible={visibleCount > i}
                animDone={barsVisible}
              />
            ))}
          </div>
        )}
      </div>

      <style>{`
        @keyframes ragWave {
          from { transform: scaleY(0.4); opacity: 0.2; }
          to   { transform: scaleY(1.6); opacity: 0.8; }
        }
      `}</style>
    </div>
  )
}
