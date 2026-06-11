import { useEffect, useState } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, Legend
} from 'recharts'
import { api } from '../api.js'

const MONTHS = ['Jan','Feb','Mar','Apr','May','Jun',
                 'Jul','Aug','Sep','Oct','Nov','Dec']

// Simulated monthly interaction data (festival spikes in Oct/Nov)
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
  { month: 'Oct', interactions: 5280, festival: 'Dashain 🎉' },
  { month: 'Nov', interactions: 4950, festival: 'Tihar ✨' },
  { month: 'Dec', interactions: 3697 },
]

const CAT_COLORS = ['#F85606','#2979FF','#00C853','#FFB300','#9C27B0','#00BCD4']

const RFM_SEGMENTS = [
  { segment: 'Champions',  color: '#856404', bg: '#FFF3CD', desc: 'High value, frequent buyers' },
  { segment: 'Loyal',      color: '#065F46', bg: '#D1FAE5', desc: 'Regular customers' },
  { segment: 'Potential',  color: '#1E40AF', bg: '#DBEAFE', desc: 'Growing engagement' },
  { segment: 'At Risk',    color: '#9F1239', bg: '#FFE4E6', desc: 'Declining activity' },
]

function StatCard({ icon, label, value, sub, color }) {
  return (
    <div className="stat-card">
      <div className="stat-icon">{icon}</div>
      <div className="stat-label">{label}</div>
      <div className="stat-value" style={{ color: color || 'var(--text)' }}>{value}</div>
      <div className="stat-sub">{sub}</div>
    </div>
  )
}

const CustomBar = (props) => {
  const { x, y, width, height, festival } = props
  return (
    <rect
      x={x} y={y} width={width} height={height}
      fill={festival ? '#F85606' : '#BBDEFB'}
      rx={4}
    />
  )
}

export default function Dashboard() {
  const [stats, setStats]   = useState(null)
  const [cats,  setCats]    = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    Promise.all([api.stats(), api.categories()])
      .then(([s, c]) => {
        setStats(s)
        setCats(c.categories || [])
      })
      .catch(console.error)
      .finally(() => setLoading(false))
  }, [])

  if (loading) return (
    <div className="loading"><div className="spinner" /><p>Loading dashboard…</p></div>
  )

  const pieData = cats.map(c => ({ name: c.name, value: c.product_count }))

  const rfmData = stats?.rfm_segments
    ? Object.entries(stats.rfm_segments).map(([k, v]) => ({ segment: k, count: v }))
    : []

  return (
    <div>
      <div className="page-header">
        <div className="page-title">Dashboard</div>
        <div className="page-subtitle">
          ML Recommendation System overview — Daraz Nepal
        </div>
      </div>

      {/* Stats */}
      <div className="stats-grid">
        <StatCard icon="👥" label="Total Users"     value={stats?.total_users?.toLocaleString()}
                  sub="Young adults (18–30)" color="var(--orange)" />
        <StatCard icon="📦" label="Products"        value={stats?.total_products?.toLocaleString()}
                  sub={`${stats?.categories} categories`} />
        <StatCard icon="🛒" label="Interactions"    value={stats?.total_interactions?.toLocaleString()}
                  sub="Views, wishlist, purchases" />
        <StatCard icon="💳" label="Conversion Rate" value={`${(stats?.conversion_rate * 100).toFixed(2)}%`}
                  sub="View → Purchase" color="var(--green)" />
      </div>

      {/* Charts */}
      <div className="charts-row">
        {/* Monthly interactions */}
        <div className="chart-card">
          <div className="chart-title">
            Monthly Interactions
            <span style={{ fontSize:11, color:'var(--muted)', marginLeft:8 }}>
              🟠 Festival months (Dashain / Tihar)
            </span>
          </div>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={MONTHLY_DATA} barSize={28}>
              <XAxis dataKey="month" tick={{ fontSize: 12 }} axisLine={false} tickLine={false} />
              <YAxis tick={{ fontSize: 11 }} axisLine={false} tickLine={false}
                     tickFormatter={v => (v/1000).toFixed(1)+'k'} />
              <Tooltip
                formatter={(v, n, p) => [
                  v.toLocaleString(),
                  p.payload.festival ? `Interactions (${p.payload.festival})` : 'Interactions'
                ]}
              />
              <Bar dataKey="interactions" shape={<CustomBar />} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Category distribution */}
        <div className="chart-card">
          <div className="chart-title">Product Categories</div>
          <ResponsiveContainer width="100%" height={220}>
            <PieChart>
              <Pie data={pieData} cx="50%" cy="50%" outerRadius={75}
                   dataKey="value" nameKey="name" label={false}>
                {pieData.map((_, i) => (
                  <Cell key={i} fill={CAT_COLORS[i % CAT_COLORS.length]} />
                ))}
              </Pie>
              <Legend iconSize={10} iconType="circle"
                      formatter={(v) => <span style={{ fontSize: 11 }}>{v}</span>} />
              <Tooltip formatter={(v, n) => [v + ' products', n]} />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Second row */}
      <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:16 }}>

        {/* Interaction breakdown */}
        <div className="chart-card">
          <div className="chart-title">Interaction Funnel</div>
          <div style={{ display:'flex', flexDirection:'column', gap:12 }}>
            {[
              { label:'Views',     value: stats?.total_views,     pct: 84.4, color:'#BBDEFB' },
              { label:'Wishlists', value: stats?.total_wishlists, pct: 9.3,  color:'#FFB300' },
              { label:'Purchases', value: stats?.total_purchases, pct: 6.2,  color:'#F85606' },
            ].map(row => (
              <div key={row.label}>
                <div style={{ display:'flex', justifyContent:'space-between',
                              fontSize:13, marginBottom:4 }}>
                  <span style={{ fontWeight:600 }}>{row.label}</span>
                  <span style={{ color:'var(--muted)' }}>
                    {row.value?.toLocaleString()} ({row.pct}%)
                  </span>
                </div>
                <div className="score-bar">
                  <div className="score-fill"
                       style={{ width:`${row.pct}%`, background: row.color }} />
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* RFM segments */}
        <div className="chart-card">
          <div className="chart-title">RFM Customer Segments</div>
          <table className="rfm-table">
            <thead>
              <tr>
                <th>Segment</th>
                <th>Users</th>
                <th>Description</th>
              </tr>
            </thead>
            <tbody>
              {RFM_SEGMENTS.map(s => {
                const count = stats?.rfm_segments?.[s.segment] ?? '–'
                return (
                  <tr key={s.segment}>
                    <td>
                      <span className="segment-badge"
                            style={{ background: s.bg, color: s.color }}>
                        {s.segment}
                      </span>
                    </td>
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
      <div className="chart-card" style={{ marginTop: 16, padding: 20 }}>
        <div className="chart-title">🎉 Nepal Festival Shopping Insights</div>
        <div style={{ display:'grid', gridTemplateColumns:'repeat(3,1fr)', gap:16 }}>
          {[
            { festival:'Dashain (Oct)', interactions: stats?.festival_interactions ?? 0,
              icon:'🪁', peak:'Electronics spike (+180%)', color:'#F85606' },
            { festival:'Tihar (Nov)',   interactions: Math.round((stats?.festival_interactions ?? 0)*0.45),
              icon:'🪔', peak:'Fashion spike (+220%)',     color:'#FFB300' },
            { festival:'Regular months', interactions: stats?.total_interactions ?? 0,
              icon:'📅', peak:'Avg 3,750 interactions/month', color:'#2979FF' },
          ].map(f => (
            <div key={f.festival} style={{ padding:16, background:'var(--bg)',
                                           borderRadius:10, borderLeft:`4px solid ${f.color}` }}>
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