import { useState } from 'react'
import Dashboard          from './pages/Dashboard.jsx'
import Recommendations    from './pages/Recommendations.jsx'
import Products           from './pages/Products.jsx'
import ModelMetrics       from './pages/ModelMetrics.jsx'
import EthicsAndColdStart from './pages/EthicsAndColdStart.jsx'

const PAGES = [
  { id: 'dashboard',       label: 'Dashboard',        icon: '📊' },
  { id: 'recommendations', label: 'Recommendations',  icon: '🎯' },
  { id: 'products',        label: 'Products',         icon: '📦' },
  { id: 'metrics',         label: 'Model Evaluation', icon: '📈' },
  { id: 'ethics',          label: 'Ethics & Cold-Start', icon: '🛡️' },
]

export default function App() {
  const [page, setPage] = useState('dashboard')
  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="sidebar-logo">
          <span>daraz</span>
          <small>ML Recommendation System</small>
        </div>
        <div className="nav-section">
          <div className="nav-label">Navigation</div>
          {PAGES.map(p => (
            <div key={p.id}
                 className={`nav-item ${page === p.id ? 'active' : ''}`}
                 onClick={() => setPage(p.id)}>
              <span className="nav-icon">{p.icon}</span>
              {p.label}
            </div>
          ))}
        </div>
        <div className="sidebar-footer">
          Binnol Dahal · 14809734<br />
          Coventry University
        </div>
      </aside>
      <main className="main">
        {page === 'dashboard'       && <Dashboard />}
        {page === 'recommendations' && <Recommendations />}
        {page === 'products'        && <Products />}
        {page === 'metrics'         && <ModelMetrics />}
        {page === 'ethics'          && <EthicsAndColdStart />}
      </main>
    </div>
  )
}