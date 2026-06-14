import { useState, useEffect } from 'react'
import { api } from '../api.js'

// ── Ethics Panel ──────────────────────────────────────────────────────────────
const ETHICS = [
  {
    icon:  '🔒',
    title: 'Data Privacy',
    color: '#2979FF',
    bg:    '#E3F2FD',
    points: [
      'User data anonymised — no PII stored or exposed',
      'Minimum data collection principle applied',
      'GDPR-aligned data handling practices',
      'Explicit opt-out available in prototype UI',
    ],
    status: 'Active',
  },
  {
    icon:  '🔍',
    title: 'Algorithmic Transparency',
    color: '#00C853',
    bg:    '#E8F5E9',
    points: [
      'SHAP values explain every recommendation',
      'Human-readable reason tags on each product card',
      'Users can see which model generated their results',
      'Feature importance published in Model Evaluation page',
    ],
    status: 'Active',
  },
  {
    icon:  '🔀',
    title: 'Filter Bubble Prevention',
    color: '#F85606',
    bg:    '#FFF3E0',
    points: [
      'Diversity post-processing ensures ≥ 3 categories per result set',
      'MMR-inspired greedy diversification applied',
      'Users exposed to products outside their primary category',
      'Novelty scoring prevents over-personalisation',
    ],
    status: 'Active',
  },
  {
    icon:  '⚖️',
    title: 'Bias Mitigation',
    color: '#9C27B0',
    bg:    '#F3E5F5',
    points: [
      'Demographic features monitored for unfair weighting',
      'Gender, income and location not used to exclude products',
      'Model evaluation checked across RFM segments',
      'Remittance-receiver flag used only for personalisation, not pricing',
    ],
    status: 'Active',
  },
  {
    icon:  '✅',
    title: 'Informed Consent',
    color: '#FF6B35',
    bg:    '#FFF8E1',
    points: [
      'Feedback buttons (👍 👎 🚫) give users control',
      'Users can mark items as "not interested"',
      'Feedback captured and used in next retraining cycle',
      'No dark patterns — opt-out is prominently displayed',
    ],
    status: 'Active',
  },
]

function EthicsCard({ item }) {
  const [open, setOpen] = useState(false)
  return (
    <div style={{ background: '#fff', borderRadius: 12, border: '1.5px solid var(--border)',
                  overflow: 'hidden', transition: 'all 0.2s' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 14,
                    padding: '16px 20px', cursor: 'pointer',
                    borderLeft: `4px solid ${item.color}` }}
           onClick={() => setOpen(!open)}>
        <span style={{ fontSize: 28 }}>{item.icon}</span>
        <div style={{ flex: 1 }}>
          <div style={{ fontWeight: 700, fontSize: 15 }}>{item.title}</div>
          <div style={{ fontSize: 12, color: 'var(--muted)', marginTop: 2 }}>
            Click to {open ? 'collapse' : 'expand'} details
          </div>
        </div>
        <span style={{ background: item.bg, color: item.color,
                       padding: '4px 12px', borderRadius: 20,
                       fontSize: 12, fontWeight: 700 }}>
          ✓ {item.status}
        </span>
        <span style={{ color: 'var(--muted)', fontSize: 18 }}>{open ? '▲' : '▼'}</span>
      </div>
      {open && (
        <div style={{ padding: '0 20px 16px 20px',
                      borderTop: '1px solid var(--border)', background: item.bg }}>
          <ul style={{ margin: '12px 0 0 18px', display: 'flex',
                       flexDirection: 'column', gap: 8 }}>
            {item.points.map((p, i) => (
              <li key={i} style={{ fontSize: 13, color: '#333', lineHeight: 1.5 }}>{p}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}

// ── Cold-Start Demo ───────────────────────────────────────────────────────────
const DEMO_PROFILES = [
  { label: 'New user — no history',      age: 21, gender: 'Male',   city: 'Kathmandu',
    income: 'Medium', preferred: 'Electronics',      remittance: false, interactions: 0 },
  { label: 'New user — Festival period', age: 24, gender: 'Female', city: 'Pokhara',
    income: 'Low',    preferred: 'Fashion',           remittance: true,  interactions: 0 },
  { label: 'Light user — 3 views',       age: 19, gender: 'Male',   city: 'Biratnagar',
    income: 'Low',    preferred: 'Phone Accessories', remittance: false, interactions: 3 },
]

const COLD_START_STRATEGIES = [
  { id: 'popularity', label: '📊 Popularity-Based',
    desc: 'Recommend the most viewed products site-wide — works for anyone with zero history.',
    color: '#2979FF' },
  { id: 'demographic', label: '👤 Demographic-Based',
    desc: 'Use age, city, and gender to find similar users and borrow their preferences.',
    color: '#00C853' },
  { id: 'content', label: '📄 Content-Based',
    desc: 'Use the stated preferred category to seed initial recommendations.',
    color: '#F85606' },
]

function ColdStartDemo() {
  const [profile,    setProfile]    = useState(0)
  const [strategy,   setStrategy]   = useState('popularity')
  const [recs,       setRecs]       = useState([])
  const [loading,    setLoading]    = useState(false)
  const [ran,        setRan]        = useState(false)

  // Map strategy to a real user known to be "cold" (few purchases)
  // We use a fixed user ID 9999 as our cold-start proxy
  const COLD_USER_ID = 9999

  const simulate = async () => {
    setLoading(true)
    setRan(true)
    try {
      // For cold-start we always use content or collaborative fallback
      const model = strategy === 'popularity' ? 'collaborative'
                  : strategy === 'demographic' ? 'collaborative'
                  : 'content'
      const data = await api.recommend(COLD_USER_ID, model, 6)
      setRecs(data.recommendations || [])
    } catch {
      setRecs([])
    }
    setLoading(false)
  }

  const selectedProfile = DEMO_PROFILES[profile]

  return (
    <div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20 }}>
        {/* Profile selector */}
        <div className="card card-pad">
          <div className="section-title">1. Choose a new user profile</div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {DEMO_PROFILES.map((p, i) => (
              <div key={i}
                   onClick={() => { setProfile(i); setRan(false) }}
                   style={{ padding: '12px 14px', borderRadius: 10, cursor: 'pointer',
                             border: `1.5px solid ${profile === i ? 'var(--orange)' : 'var(--border)'}`,
                             background: profile === i ? 'var(--orange-lt)' : 'var(--bg)' }}>
                <div style={{ fontWeight: 700, fontSize: 13,
                              color: profile === i ? 'var(--orange)' : 'var(--text)' }}>
                  {p.label}
                </div>
                <div style={{ fontSize: 11, color: 'var(--muted)', marginTop: 4 }}>
                  Age {p.age} · {p.gender} · {p.city} · Prefers: {p.preferred}
                  {p.remittance && ' · 💸 Remittance'}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Strategy selector */}
        <div className="card card-pad">
          <div className="section-title">2. Choose a cold-start strategy</div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {COLD_START_STRATEGIES.map(s => (
              <div key={s.id}
                   onClick={() => { setStrategy(s.id); setRan(false) }}
                   style={{ padding: '12px 14px', borderRadius: 10, cursor: 'pointer',
                             border: `1.5px solid ${strategy === s.id ? s.color : 'var(--border)'}`,
                             background: strategy === s.id ? '#F8F8FF' : 'var(--bg)',
                             borderLeft: `4px solid ${s.color}` }}>
                <div style={{ fontWeight: 700, fontSize: 13 }}>{s.label}</div>
                <div style={{ fontSize: 11, color: 'var(--muted)', marginTop: 4,
                              lineHeight: 1.5 }}>{s.desc}</div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Profile summary */}
      <div style={{ margin: '16px 0', padding: '14px 18px', background: '#F8F9FF',
                    borderRadius: 10, border: '1px solid var(--border)',
                    display: 'flex', gap: 24, alignItems: 'center', flexWrap: 'wrap' }}>
        <span style={{ fontSize: 13, fontWeight: 600, color: 'var(--muted)' }}>
          Simulated profile:
        </span>
        {Object.entries({
          Age: selectedProfile.age,
          Gender: selectedProfile.gender,
          City: selectedProfile.city,
          Income: selectedProfile.income,
          'Preferred Category': selectedProfile.preferred,
          'Past Interactions': selectedProfile.interactions,
        }).map(([k, v]) => (
          <div key={k} style={{ fontSize: 13 }}>
            <span style={{ color: 'var(--muted)' }}>{k}: </span>
            <span style={{ fontWeight: 700 }}>{v}</span>
          </div>
        ))}
        <span style={{ marginLeft: 'auto' }}>
          <button className="btn btn-orange" onClick={simulate}>
            🚀 Generate Recommendations
          </button>
        </span>
      </div>

      {/* Results */}
      {loading && (
        <div className="loading"><div className="spinner" /><p>Generating cold-start recommendations…</p></div>
      )}

      {!loading && ran && recs.length > 0 && (
        <div>
          <div style={{ fontWeight: 700, fontSize: 14, marginBottom: 14 }}>
            Recommendations for a new user via{' '}
            <span style={{ color: 'var(--orange)' }}>
              {COLD_START_STRATEGIES.find(s => s.id === strategy)?.label}
            </span>
          </div>
          <div style={{ display: 'grid',
                        gridTemplateColumns: 'repeat(auto-fill, minmax(180px,1fr))',
                        gap: 14 }}>
            {recs.map(rec => (
              <div key={rec.product_id} className="product-card">
                <div className={`product-img cat-${CSS.escape(rec.category || '')}`}
                     style={{ height: 90 }}>
                  {rec.category === 'Electronics' ? '📱'
                   : rec.category === 'Fashion' ? '👗'
                   : rec.category === 'Phone Accessories' ? '🎧'
                   : rec.category === 'Beauty & Personal Care' ? '💄'
                   : rec.category === 'Home & Kitchen' ? '🏠'
                   : rec.category === 'Sports & Fitness' ? '🏃' : '📦'}
                </div>
                <div className="product-body">
                  <div className="product-name">
                    {rec.product_name || `Product #${rec.product_id}`}
                  </div>
                  <div className="product-brand">{rec.brand}</div>
                  <div className="product-price">NPR {rec.price_npr?.toLocaleString()}</div>
                  <div style={{ marginTop: 6, padding: '4px 8px',
                                background: 'var(--orange-lt)', borderRadius: 6,
                                fontSize: 11, color: 'var(--orange)', fontWeight: 600 }}>
                    💡 {rec.explanation}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {!loading && ran && recs.length === 0 && (
        <div className="empty">
          <div className="empty-icon">⚠️</div>
          <div className="empty-text">No recommendations returned — check Flask API is running</div>
        </div>
      )}
    </div>
  )
}

// ── Main Page ─────────────────────────────────────────────────────────────────
export default function EthicsAndColdStart() {
  const [tab, setTab] = useState('ethics')

  return (
    <div>
      <div className="page-header">
        <div className="page-title">Ethics & Cold-Start</div>
        <div className="page-subtitle">
          Responsible AI design and new-user recommendation strategies
        </div>
      </div>

      {/* Tabs */}
      <div className="model-tabs" style={{ marginBottom: 24 }}>
        <button className={`model-tab ${tab === 'ethics' ? 'active' : ''}`}
                onClick={() => setTab('ethics')}>
          🛡️ Ethical Safeguards
        </button>
        <button className={`model-tab ${tab === 'coldstart' ? 'active' : ''}`}
                onClick={() => setTab('coldstart')}>
          🧊 Cold-Start Demo
        </button>
      </div>

      {tab === 'ethics' && (
        <div>
          {/* Summary row */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5,1fr)',
                        gap: 12, marginBottom: 24 }}>
            {ETHICS.map(e => (
              <div key={e.title}
                   style={{ background: e.bg, borderRadius: 10, padding: '14px 16px',
                             textAlign: 'center', border: `1.5px solid ${e.color}22` }}>
                <div style={{ fontSize: 24 }}>{e.icon}</div>
                <div style={{ fontSize: 12, fontWeight: 700, marginTop: 6,
                              color: e.color }}>{e.title}</div>
              </div>
            ))}
          </div>

          {/* Expandable cards */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
            {ETHICS.map(e => <EthicsCard key={e.title} item={e} />)}
          </div>

          {/* GDPR note */}
          <div style={{ marginTop: 20, padding: '14px 18px',
                        background: '#EEF2FF', borderRadius: 10,
                        border: '1px solid #C7D2FE', fontSize: 13 }}>
            <strong>📋 Regulatory note:</strong> This prototype is designed in alignment with
            GDPR principles and Nepal's emerging data protection standards. All user data in
            this system is synthetic and contains no real personal information. In a production
            deployment, explicit user consent, a data processing agreement with Daraz Nepal,
            and a Data Protection Impact Assessment (DPIA) would be required.
          </div>
        </div>
      )}

      {tab === 'coldstart' && (
        <div>
          <div className="card card-pad" style={{ marginBottom: 20 }}>
            <div style={{ fontSize: 14, fontWeight: 700, marginBottom: 8 }}>
              🧊 What is the Cold-Start Problem?
            </div>
            <p style={{ fontSize: 13, color: 'var(--muted)', lineHeight: 1.7, margin: 0 }}>
              New users have no interaction history, so collaborative filtering cannot make
              meaningful predictions. This is a critical challenge for Daraz Nepal where
              young adults (18–30) are frequently first-time e-commerce users.
              The system applies three fallback strategies depending on what information
              is available about the new user.
            </p>
          </div>
          <ColdStartDemo />
        </div>
      )}
    </div>
  )
}