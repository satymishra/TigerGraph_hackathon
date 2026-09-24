import { Routes, Route, NavLink } from 'react-router-dom'
import { LayoutDashboard, Search, Zap, GitBranch, GitCompare } from 'lucide-react'
import Dashboard from './pages/Dashboard'
import QuestionExplorer from './pages/QuestionExplorer'
import LiveQuery from './pages/LiveQuery'
import Architecture from './pages/Architecture'
import Compare from './pages/Compare'

const NAV = [
  { to: '/',            icon: LayoutDashboard, label: 'Dashboard'   },
  { to: '/explore',     icon: Search,          label: 'Questions'   },
  { to: '/query',       icon: Zap,             label: 'Live Query'  },
  { to: '/compare',     icon: GitCompare,      label: 'Compare'     },
  { to: '/architecture',icon: GitBranch,       label: 'Architecture'},
]

export default function App() {
  return (
    <div className="flex h-screen overflow-hidden">
      {/* Sidebar */}
      <aside className="w-56 flex-shrink-0 bg-surface-800 border-r border-surface-600 flex flex-col">
        {/* Logo */}
        <div className="px-5 py-5 border-b border-surface-600">
          <div className="flex items-center gap-2.5">
            <div className="w-7 h-7 rounded-lg bg-brand-500 flex items-center justify-center text-white font-bold text-sm">G</div>
            <div>
              <div className="text-sm font-semibold text-white leading-tight">GraphRAG</div>
              <div className="text-[10px] text-surface-500 leading-tight">Olympic Benchmark</div>
            </div>
          </div>
        </div>

        {/* Nav */}
        <nav className="flex-1 py-4 px-3 space-y-0.5">
          {NAV.map(({ to, icon: Icon, label }) => (
            <NavLink
              key={to}
              to={to}
              end={to === '/'}
              className={({ isActive }) =>
                `flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-all duration-150 ${
                  isActive
                    ? 'bg-brand-500/15 text-brand-400 border border-brand-500/30'
                    : 'text-gray-400 hover:text-gray-200 hover:bg-surface-700'
                }`
              }
            >
              <Icon size={16} />
              {label}
            </NavLink>
          ))}
        </nav>

        {/* Footer */}
        <div className="px-4 py-4 border-t border-surface-600">
          <div className="text-[11px] text-surface-500 leading-relaxed">
            TigerGraph Hackathon<br />
            <span className="text-brand-400">Round 1 · Sep 2026</span>
          </div>
        </div>
      </aside>

      {/* Main content */}
      <main className="flex-1 overflow-y-auto bg-surface-900">
        <Routes>
          <Route path="/"             element={<Dashboard />}        />
          <Route path="/explore"      element={<QuestionExplorer />} />
          <Route path="/query"        element={<LiveQuery />}        />
          <Route path="/compare"      element={<Compare />}          />
          <Route path="/architecture" element={<Architecture />}     />
        </Routes>
      </main>
    </div>
  )
}
