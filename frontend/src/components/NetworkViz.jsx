import { useEffect, useRef, useState, useCallback } from 'react'
import cytoscape from 'cytoscape'
import { ZoomIn, ZoomOut, Maximize2, X } from 'lucide-react'

// ── Timeline layout: X = year, Y = sport rank ────────────────
function computeLayout(nodes, W, H) {
  const years  = [...new Set(nodes.map(n => n.year).filter(Boolean))].sort((a,b)=>a-b)
  const sports = [...new Set(nodes.map(n => n.sport || 'Other'))].sort()
  const yMin = years[0] ?? 2000
  const yMax = years[years.length - 1] ?? 2022
  const yRange = yMax - yMin || 1

  const positions = {}
  nodes.forEach((n, i) => {
    const xFrac = (n.year - yMin) / yRange
    const yFrac = sports.length > 1
      ? sports.indexOf(n.sport || 'Other') / (sports.length - 1)
      : 0.5
    // Deterministic jitter so layout is stable across renders
    const jx = ((i * 41 + 7)  % 60) - 30
    const jy = ((i * 29 + 11) % 44) - 22
    positions[n.id] = {
      x: 70 + xFrac * (W - 140) + jx,
      y: 40 + yFrac * (H - 80)  + jy,
    }
  })
  return positions
}

// ── Cytoscape stylesheet ──────────────────────────────────────
const STYLE = [
  {
    selector: 'node',
    style: {
      'background-color': '#0e2444',
      'width':  12,
      'height': 12,
      'border-width':  1.2,
      'border-color':  '#1e4a88',
      'label': '',
      'opacity': 0.85,
      'shadow-blur': 0,
    },
  },
  {
    // Scanned: visible blue flash during rapid scan phase
    selector: 'node.scanned',
    style: {
      'background-color': '#0f5a8a',
      'border-color':     '#1e8abf',
      'border-width':     1.8,
      'width':  13,
      'height': 13,
      'opacity': 1,
      'z-index': 10,
      'shadow-blur':    12,
      'shadow-color':   '#00c8ff',
      'shadow-opacity': 0.5,
      'shadow-offset-x': 0,
      'shadow-offset-y': 0,
    },
  },
  {
    // Lit: brighter cyan for traversal path edges (kept for edge highlighting)
    selector: 'node.lit',
    style: {
      'background-color': '#00c8ff',
      'border-color':     '#66e8ff',
      'border-width':     2,
      'width':  13,
      'height': 13,
      'opacity': 1,
      'z-index': 15,
      'shadow-blur':    14,
      'shadow-color':   '#00c8ff',
      'shadow-opacity': 1,
      'shadow-offset-x': 0,
      'shadow-offset-y': 0,
    },
  },
  {
    // Answer: gold glow + event name label on the final node
    selector: 'node.answer',
    style: {
      'background-color': '#f59e0b',
      'border-color':     '#fde68a',
      'border-width':     2.5,
      'width':  16,
      'height': 16,
      'opacity': 1,
      'z-index': 25,
      'shadow-blur':    22,
      'shadow-color':   '#f59e0b',
      'shadow-opacity': 1,
      'shadow-offset-x': 0,
      'shadow-offset-y': 0,
      'label':                   'data(shortLabel)',
      'color':                   '#fde68a',
      'font-size':               '7px',
      'font-family':             'monospace',
      'text-valign':             'bottom',
      'text-margin-y':           5,
      'text-background-color':   '#060c14',
      'text-background-opacity': 0.92,
      'text-background-padding': '2px',
    },
  },
  {
    // Signal dot traveling along edge
    selector: 'node.__signal',
    style: {
      'background-color': '#ffffff',
      'border-width':  0,
      'width':  9,
      'height': 9,
      'opacity': 1,
      'z-index': 100,
      'shadow-blur':    18,
      'shadow-color':   '#00c8ff',
      'shadow-opacity': 1,
      'shadow-offset-x': 0,
      'shadow-offset-y': 0,
    },
  },
  {
    selector: 'node.selected',
    style: {
      'border-width':   3,
      'border-color':   '#ffffff',
      'shadow-blur':    12,
      'shadow-color':   '#ffffff',
      'shadow-opacity': 0.6,
    },
  },
  {
    selector: 'edge',
    style: {
      'curve-style':         'straight',
      'width':               1.4,
      'line-color':          '#1e4a88',
      'target-arrow-shape':  'none',
      'opacity':             0.55,
    },
  },
  {
    selector: 'edge.lit',
    style: {
      'width':   2.8,
      'line-color': '#00c8ff',
      'opacity': 1,
      'z-index': 12,
      'shadow-blur':    10,
      'shadow-color':   '#00c8ff',
      'shadow-opacity': 0.9,
    },
  },
  {
    selector: 'edge.dimmed',
    style: { 'opacity': 0.06 },
  },
]

// ── Async traversal animation ─────────────────────────────────
// cancelRef is a per-instance { current: bool } ref — no module-level shared flag
async function runTraversal(cy, traversedIds, answerIds, onStep, onDone, cancelRef) {
  cancelRef.current = false
  if (!traversedIds.length) { onDone?.(); return }

  const useSignals = traversedIds.length <= 6   // signal dots for small traversals

  // ── Phase 1a (small): signal dot hops between nodes ─────────
  if (useSignals) {
    const first = cy.getElementById(traversedIds[0])
    if (first.length) { first.addClass('scanned'); onStep(0) }

    for (let i = 1; i < traversedIds.length; i++) {
      if (cancelRef.current) return

      const srcId   = traversedIds[i - 1]
      const tgtId   = traversedIds[i]
      const srcNode = cy.getElementById(srcId)
      const tgtNode = cy.getElementById(tgtId)
      if (!tgtNode.length) { onStep(i); continue }

      const sigId = `__sig_${i}_${Date.now()}`
      cy.add({ data: { id: sigId }, classes: '__signal',
               position: srcNode.length ? srcNode.position() : tgtNode.position() })

      await new Promise(resolve => {
        if (cancelRef.current) { safeRemove(cy, sigId); resolve(); return }
        cy.getElementById(sigId).animate(
          { position: tgtNode.position() },
          { duration: 180, easing: 'ease-in-out',
            complete: () => { safeRemove(cy, sigId); resolve() } }
        )
      })
      if (cancelRef.current) return

      cy.edges(`[source="${srcId}"][target="${tgtId}"],[source="${tgtId}"][target="${srcId}"]`).addClass('lit')
      tgtNode.addClass('scanned')
      onStep(i)
      await sleep(40)
    }

  // ── Phase 1b (large): overlapping signals — "neural rain" ───
  } else {
    for (let i = 0; i < traversedIds.length; i++) {
      if (cancelRef.current) return
      const node = cy.getElementById(traversedIds[i])
      if (node.length) node.addClass('scanned')

      if (i > 0) {
        const src     = traversedIds[i - 1]
        const tgt     = traversedIds[i]
        const srcNode = cy.getElementById(src)
        const tgtNode = cy.getElementById(tgt)

        cy.edges(`[source="${src}"][target="${tgt}"],[source="${tgt}"][target="${src}"]`).addClass('lit')

        if (srcNode.length && tgtNode.length) {
          const sigId = `__sig_${i}_${Date.now()}`
          try {
            cy.add({ data: { id: sigId }, classes: '__signal',
                     position: { x: srcNode.position().x, y: srcNode.position().y } })
            cy.getElementById(sigId).animate(
              { position: { x: tgtNode.position().x, y: tgtNode.position().y } },
              { duration: 110, easing: 'ease-in-out',
                complete: () => safeRemove(cy, sigId) }
            )
          } catch (_) { safeRemove(cy, sigId) }
        }
      }

      onStep(i)
      await sleep(38)
    }
  }

  if (cancelRef.current) return
  await sleep(250)

  // ── Phase 2: answer nodes turn gold ─────────────────────────
  for (const id of answerIds) {
    if (cancelRef.current) return
    const node = cy.getElementById(id)
    if (node.length) node.removeClass('scanned lit').addClass('answer')
    await sleep(120)
  }

  // ── Phase 3: fit view around the path ───────────────────────
  await sleep(320)
  if (cancelRef.current) return
  const pathNodes = cy.nodes().filter(n => traversedIds.includes(n.id()))
  if (pathNodes.length > 0) {
    cy.animate({
      fit: { eles: pathNodes.neighborhood().add(pathNodes), padding: 80 },
      duration: 600,
    })
  }

  await sleep(720)
  if (!cancelRef.current) onDone?.()
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)) }
function safeRemove(cy, id) { try { cy.getElementById(id).remove() } catch {} }

// ── Sport colour (subtle, same-ish palette) ───────────────────
function sportColor() { return '#0e2444' }  // all nodes same dim blue

// ── Component ─────────────────────────────────────────────────
export default function NetworkViz({
  running, done,
  traversedIds    = [],
  traversedNodes  = [],
  answerIds       = [],
  onAnimationDone,
  height,
}) {
  const W = 920, H = height ?? 480
  const containerRef   = useRef(null)
  const cyRef          = useRef(null)
  const cancelRef      = useRef(false)
  const [graphData,    setGraphData]   = useState(null)
  const [loadingGraph, setLoadingGraph] = useState(true)
  const [cyReady,      setCyReady]     = useState(false)
  const [selectedNode, setSelectedNode] = useState(null)
  const [step,         setStep]         = useState(-1)

  // ── Load graph data ───────────────────────────────────────
  useEffect(() => {
    fetch('/api/graph?limit=350')
      .then(r => r.json())
      .then(d => { setGraphData(d); setLoadingGraph(false) })
      .catch(() => setLoadingGraph(false))
  }, [])

  // ── Build Cytoscape instance ──────────────────────────────
  useEffect(() => {
    if (!containerRef.current || !graphData?.nodes?.length || cyRef.current) return

    const positions = computeLayout(graphData.nodes, W, H)

    const elements = [
      ...graphData.nodes.map(n => ({
        data: {
          id:         n.id,
          shortLabel: (n.label || n.id).slice(0, 20),
          fullLabel:  n.label || n.id,
          sport:      n.sport,
          year:       n.year,
          season:     n.season,
          venue:      n.venue,
          competitors: n.competitors,
          nations:    n.nations,
          date_str:   n.date_str,
        },
        position: positions[n.id] || { x: W / 2, y: H / 2 },
      })),
      ...graphData.edges.map((e, i) => ({
        data: { id: `e${i}`, source: e.source, target: e.target, type: e.type },
      })),
    ]

    const cy = cytoscape({
      container: containerRef.current,
      elements,
      style: STYLE,
      layout: { name: 'preset' },
      userZoomingEnabled:  true,
      userPanningEnabled:  true,
      boxSelectionEnabled: false,
      minZoom: 0.08,
      maxZoom: 8,
      wheelSensitivity: 0.25,
    })

    // Dark canvas background
    try { cy.renderer().data.canvasContainer.style.background = '#060c14' } catch {}

    cy.on('tap', 'node', evt => {
      if (evt.target.hasClass('__signal')) return
      setSelectedNode(evt.target.data())
      cy.nodes().removeClass('selected')
      evt.target.addClass('selected')
    })
    cy.on('tap', evt => {
      if (evt.target === cy) { setSelectedNode(null); cy.nodes().removeClass('selected') }
    })

    cyRef.current = cy
    cy.fit(undefined, 30)
    setCyReady(true)
  }, [graphData])

  // ── Trigger traversal animation ───────────────────────────
  useEffect(() => {
    const cy = cyRef.current
    if (!cy) return

    cancelRef.current = true
    setStep(-1)

    // Remove any in-flight signal dots and clear traversal highlights
    cy.$('.__signal').remove()
    cy.elements().removeClass('lit answer scanned')

    // Only exit early without animation if not yet done
    if (!done) return
    // Done but nothing to animate — unblock the answer immediately
    if (!traversedIds.length) { onAnimationDone?.(); return }

    // Inject any traversed nodes missing from the graph
    traversedNodes.forEach(nd => {
      if (!cy.getElementById(nd.id).length) {
        const positions = computeLayout(graphData?.nodes ?? [], W, H)
        const pos = positions[nd.id] || {
          x: W / 2 + (Math.random() - .5) * 160,
          y: H / 2 + (Math.random() - .5) * 100,
        }
        cy.add({
          data: {
            id: nd.id, shortLabel: (nd.label||nd.id).slice(0,20),
            fullLabel: nd.label||nd.id, sport: nd.sport, year: nd.year,
            season: nd.season, venue: nd.venue, competitors: nd.competitors,
            nations: nd.nations, date_str: nd.date_str,
          },
          position: pos,
        })
      }
    })

    // Small delay so React has settled, then animate
    const t = setTimeout(() => {
      runTraversal(cy, traversedIds, answerIds, i => setStep(i), onAnimationDone, cancelRef)
    }, 120)
    return () => { cancelRef.current = true; clearTimeout(t) }

  }, [done, running, cyReady, JSON.stringify(traversedIds)])

  // ── Zoom helpers ──────────────────────────────────────────
  const zoomIn  = useCallback(() => {
    const cy = cyRef.current; if (!cy) return
    cy.zoom({ level: cy.zoom() * 1.35,
              renderedPosition: { x: cy.width()/2, y: cy.height()/2 } })
  }, [])
  const zoomOut = useCallback(() => {
    const cy = cyRef.current; if (!cy) return
    cy.zoom({ level: cy.zoom() * 0.74,
              renderedPosition: { x: cy.width()/2, y: cy.height()/2 } })
  }, [])
  const fitAll  = useCallback(() => { cyRef.current?.fit(undefined, 30) }, [])

  // ── Render ────────────────────────────────────────────────
  return (
    <div className="relative w-full rounded-xl overflow-hidden border border-surface-700"
         style={{ height: height ?? 480, background: '#060c14' }}>

      {/* Cytoscape canvas fills the whole box */}
      <div ref={containerRef} style={{ width: '100%', height: '100%' }} />

      {/* Loading */}
      {loadingGraph && (
        <div className="absolute inset-0 flex items-center justify-center"
             style={{ background: '#060c14' }}>
          <p className="text-xs text-gray-500 animate-pulse">Loading graph…</p>
        </div>
      )}

      {/* Graph stats badge */}
      {!loadingGraph && graphData && (
        <div className="absolute top-3 left-3 flex items-center gap-2 px-2.5 py-1 rounded-full"
             style={{ background: 'rgba(6,12,20,0.85)', border: '1px solid #1e3a5f' }}>
          <div className="w-1.5 h-1.5 rounded-full"
               style={{ background: running ? '#f59e0b' : done ? '#00c8ff' : '#1e4a88',
                        boxShadow: running ? '0 0 6px #f59e0b' : done ? '0 0 6px #00c8ff' : 'none' }} />
          <span className="text-[10px] text-gray-500 font-mono">
            {graphData.nodes.length} nodes · {graphData.edges.length} edges
          </span>
          {running && <span className="text-[10px] text-yellow-400 animate-pulse">traversing…</span>}
          {done && step >= 0 && (
            <span className="text-[10px] text-cyan-400">
              {Math.min(step + 1, traversedIds.length)}/{traversedIds.length} hops
            </span>
          )}
        </div>
      )}

      {/* Zoom controls */}
      <div className="absolute top-3 right-3 flex flex-col gap-1">
        {[
          { icon: <ZoomIn size={12}/>,   fn: zoomIn  },
          { icon: <ZoomOut size={12}/>,  fn: zoomOut },
          { icon: <Maximize2 size={10}/>,fn: fitAll  },
        ].map((btn, i) => (
          <button key={i} onClick={btn.fn}
                  className="w-7 h-7 flex items-center justify-center rounded text-gray-500 hover:text-white transition-colors"
                  style={{ background: 'rgba(6,12,20,0.85)', border: '1px solid #1e3a5f' }}>
            {btn.icon}
          </button>
        ))}
      </div>

      {/* Hint when idle */}
      {!loadingGraph && !running && !done && (
        <div className="absolute bottom-3 left-1/2 -translate-x-1/2 pointer-events-none">
          <span className="text-[10px] text-gray-600 px-3 py-1 rounded-full"
                style={{ background: 'rgba(6,12,20,0.85)', border: '1px solid #1e3a5f' }}>
            scroll to zoom · drag to pan · click any node
          </span>
        </div>
      )}

      {/* Node detail panel */}
      {selectedNode && !selectedNode.id?.startsWith('__') && (
        <div className="absolute bottom-3 left-3 w-56 rounded-xl p-3"
             style={{ background: 'rgba(10,18,32,0.97)', border: '1px solid #1e3a5f' }}>
          <div className="flex items-start justify-between gap-2 mb-2">
            <p className="text-xs font-semibold text-white leading-snug line-clamp-2">
              {selectedNode.fullLabel || selectedNode.shortLabel}
            </p>
            <button onClick={() => { setSelectedNode(null); cyRef.current?.nodes().removeClass('selected') }}
                    className="shrink-0 text-gray-600 hover:text-gray-400 mt-0.5">
              <X size={11}/>
            </button>
          </div>
          <div className="space-y-1">
            {[
              ['sport',       selectedNode.sport],
              ['year',        selectedNode.year],
              ['season',      selectedNode.season],
              ['venue',       selectedNode.venue],
              ['competitors', selectedNode.competitors || null],
              ['nations',     selectedNode.nations     || null],
              ['date',        selectedNode.date_str],
            ].filter(([,v]) => v).map(([k,v]) => (
              <div key={k} className="flex gap-2 text-[10px]">
                <span className="text-gray-600 w-20 shrink-0">{k}</span>
                <span className="text-gray-300">{v}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
