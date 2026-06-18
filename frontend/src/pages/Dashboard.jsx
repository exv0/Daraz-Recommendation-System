import { useEffect, useState } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, Legend
} from 'recharts'
import { api } from '../api.js'

const MONTHLY_DATA = [
  { month: 'Jan', interactions: 3929 },
  { month: 'Feb', interactions: 3651 },
  { month: 'Mar', interactions: 3818 },
  { month: 'Apr', interactions: 3698 },
  { month: 'May', interactions: 3801 },
  { month: 'Jun', interactions: 3761 },
  { month: 'Jul', interactions: 3847 },
  { month: 'Aug', interactions: 3849 },
  { month: 'Sep', interactions: 3745 },
  { month: 'Oct', interactions: 5280, festival: 'Dashain 🪁' },
  { month: 'Nov', interactions: 4950, festival: 'Tihar 🪔' },
  { month: 'Dec', interactions: 3697 },
]

const CAT_COLORS = ['#F85606','#2563EB','#059669','#D97706','#7C3AED','#0891B2']

const RFM_SEGMENTS = [
  { segment: 'Champions',  color: '#92400E', bg: '#FEF3C7', desc: 'High value, frequent buyers' },
  { segment: 'Loyal',      color: '#065F46', bg: '#D1FAE5', desc: 'Regular customers' },
  { segment: 'Potential',  color: '#1E40AF', bg: '#DBEAFE', desc: 'Growing engagement' },
  { segment: 'At Risk',    color: '#9F1239', bg: '#FFE4E6', desc: 'Declining activity' },
]

// ── Signature element: Signal Bars ──────────────────────────────────────────
function SignalBars({ level }) {
  // level: 0-4
  const cls = level >= 4 ? 'c-high' : level >= 2 ? 'c-mid' : 'c-low'
  return (
    <div className="signal-bars">
      {['b1','b2','b3','b4'].map((b, i) => (
        <div key={b} className={`signal-bar ${b} ${i < level ? `active ${cls}` : ''}`} />
      ))}
    </div>
  )
}

function StatCard({ icon, label, value, sub, color, signal }) {
  return (
    <div className={`stat-card ${color}`}>
      <div style={{ display:'flex', justifyContent:'space-between', alignItems:'flex-start' }}>
        <div className={`stat-icon-wrap ${color}`}>{icon}</div>
        {signal !== undefined && <SignalBars level={signal} />}
      </div>
      <div className="stat-label">{label}</div>
      <div className="stat-value">{value}</div>
      <div className="stat-sub">{sub}</div>
    </div>
  )
}

const CustomBar = (props) => {
  const { x, y, width, height, festival } = props
  return <rect x={x} y={y} width={width} height={height} fill={festival ? '#F85606' : '#BFDBFE'} rx={4} />
}

export default function Dashboard() {
  const [stats, setStats]   = useState(null)
  const [cats,  setCats]    = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    Promise.all([api.stats(), api.categories()])
      .then(([s, c]) => { setStats(s); setCats(c.categories || []) })
      .catch(console.error)
      .finally(() => setLoading(false))
  }, [])

  if (loading) return (
    <div className="loading"><div className="spinner" /><p>Loading dashboard…</p></div>
  )

  const pieData = cats.map(c => ({ name: c.name, value: c.product_count }))
  const conversionPct = (stats?.conversion_rate * 100) || 0
  const signalLevel = conversionPct > 8 ? 4 : conversionPct > 6 ? 3 : conversionPct > 4 ? 2 : 1

  return (
    <div>
      <div className="page-header">
        <div className="page-title">Dashboard</div>
        <div className="page-subtitle">ML Recommendation System overview — Daraz Nepal</div>
      </div>

      {/* Stats */}
      <div className="stats-grid">
        <StatCard icon="👥" label="Total Users" value={stats?.total_users?.toLocaleString()}
                  sub="Young adults (18–30)" color="orange" />
        <StatCard icon="📦" label="Products" value={stats?.total_products?.toLocaleString()}
                  sub={`${stats?.categories} categories`} color="blue" />
        <StatCard icon="🛒" label="Interactions" value={stats?.total_interactions?.toLocaleString()}
                  sub="Views, wishlist, purchases" color="purple" />
        <StatCard icon="💳" label="Conversion Rate" value={`${conversionPct.toFixed(2)}%`}
                  sub="Model confidence signal" color="green" signal={signalLevel} />
      </div>

      {/* Charts */}
      <div className="charts-row">
        <div className="chart-card">
          <div className="chart-title">
            Monthly Interactions
            <span className="chart-badge">🟠 Festival spikes</span>
          </div>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={MONTHLY_DATA} barSize={28}>
              <XAxis dataKey="month" tick={{ fontSize: 12 }} axisLine={false} tickLine={false} />
              <YAxis tick={{ fontSize: 11 }} axisLine={false} tickLine={false}
                     tickFormatter={v => (v/1000).toFixed(1)+'k'} />
              <Tooltip formatter={(v, n, p) => [
                v.toLocaleString(),
                p.payload.festival ? `Interactions (${p.payload.festival})` : 'Interactions'
              ]} />
              <Bar dataKey="interactions" shape={<CustomBar />} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="chart-card">
          <div className="chart-title">Product Categories</div>
          <ResponsiveContainer width="100%" height={220}>
            <PieChart>
              <Pie data={pieData} cx="50%" cy="50%" outerRadius={75}
                   dataKey="value" nameKey="name" label={false}>
                {pieData.map((_, i) => <Cell key={i} fill={CAT_COLORS[i % CAT_COLORS.length]} />)}
              </Pie>
              <Legend iconSize={10} iconType="circle"
                      formatter={(v) => <span style={{ fontSize: 11 }}>{v}</span>} />
              <Tooltip formatter={(v, n) => [v + ' products', n]} />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:16 }}>
        {/* Interaction funnel with signal-style bars */}
        <div className="chart-card">
          <div className="chart-title">Interaction Funnel</div>
          {[
            { label:'Views',     value: stats?.total_views,     pct: 84.4, color:'#BFDBFE' },
            { label:'Wishlists', value: stats?.total_wishlists, pct: 9.3,  color:'#FDE68A' },
            { label:'Purchases', value: stats?.total_purchases, pct: 6.2,  color:'#F85606' },
          ].map(row => (
            <div key={row.label} className="funnel-bar-wrap">
              <div className="funnel-bar-header">
                <span className="funnel-bar-name">{row.label}</span>
                <span className="funnel-bar-count">{row.value?.toLocaleString()} ({row.pct}%)</span>
              </div>
              <div className="funnel-track">
                <div className="funnel-fill" style={{ width:`${row.pct}%`, background: row.color }} />
              </div>
            </div>
          ))}
        </div>

        {/* RFM segments */}
        <div className="chart-card">
          <div className="chart-title">RFM Customer Segments</div>
          <table className="rfm-table">
            <thead><tr><th>Segment</th><th>Users</th><th>Description</th></tr></thead>
            <tbody>
              {RFM_SEGMENTS.map(s => {
                const count = stats?.rfm_segments?.[s.segment] ?? '–'
                return (
                  <tr key={s.segment}>
                    <td><span className="segment-badge" style={{ background: s.bg, color: s.color }}>{s.segment}</span></td>
                    <td style={{ fontWeight: 700 }}>{Number(count).toLocaleString()}</td>
                    <td style={{ color: 'var(--muted)', fontSize: 12 }}>{s.desc}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Festival insights */}
      <div className="chart-card" style={{ marginTop: 16 }}>
        <div className="chart-title">🎉 Nepal Festival Shopping Insights</div>
        <div style={{ display:'grid', gridTemplateColumns:'repeat(3,1fr)', gap:16 }}>
          {[
            { festival:'Dashain (Oct)', icon:'🪁', peak:'Electronics spike (+180%)', color:'#F85606', bg:'#FFF0E8' },
            { festival:'Tihar (Nov)',   icon:'🪔', peak:'Fashion spike (+220%)',     color:'#D97706', bg:'#FEF3C7' },
            { festival:'Regular months', icon:'📅', peak:'Avg 3,750 interactions/month', color:'#2563EB', bg:'#EFF6FF' },
          ].map(f => (
            <div key={f.festival} style={{ padding:16, background: f.bg, borderRadius:10, borderLeft:`4px solid ${f.color}` }}>
              <div style={{ fontSize:22, marginBottom:6 }}>{f.icon}</div>
              <div style={{ fontWeight:700, fontSize:14 }}>{f.festival}</div>
              <div style={{ fontSize:12, color:'var(--muted)', marginTop:4 }}>{f.peak}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}