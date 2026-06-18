import { useState } from 'react'
import { api } from '../api.js'

const CAT_ICONS = {
  'Electronics':             '📱',
  'Fashion':                 '👗',
  'Phone Accessories':       '🎧',
  'Beauty & Personal Care':  '💄',
  'Home & Kitchen':          '🏠',
  'Sports & Fitness':        '🏃',
}

function Stars({ rating }) {
  const full = Math.floor(rating)
  const half = rating % 1 >= 0.5
  return <span className="stars">{'★'.repeat(full)}{'☆'.repeat(5 - full - (half?1:0))}{half?'½':''}</span>
}

// ── Signature element re-used here for match confidence ────────────────────
function SignalBars({ score }) {
  const level = score >= 0.75 ? 4 : score >= 0.5 ? 3 : score >= 0.25 ? 2 : 1
  const cls = level >= 4 ? 'c-high' : level >= 2 ? 'c-mid' : 'c-low'
  return (
    <div className="signal-bars">
      {['b1','b2','b3','b4'].map((b, i) => (
        <div key={b} className={`signal-bar ${b} ${i < level ? `active ${cls}` : ''}`} />
      ))}
    </div>
  )
}

function ProductCard({ rec }) {
  const [fb, setFb] = useState(null)
  const score = rec.score || rec.hybrid_score || 0

  const handleFb = async (type) => {
    setFb(type)
    await api.feedback({ user_id: rec._userId, product_id: rec.product_id, feedback_type: type })
  }

  return (
    <div className="product-card">
      <div className={`product-img cat-${CSS.escape(rec.category || '')}`}>
        {CAT_ICONS[rec.category] || '📦'}
      </div>
      <div className="product-body">
        <div className="product-name">{rec.product_name || `Product #${rec.product_id}`}</div>
        <div className="product-brand">{rec.brand || rec.category}</div>
        <div className="product-price">
          NPR {rec.price_npr?.toLocaleString()}
          <small> / {rec.festival_relevance !== 'None' ? rec.festival_relevance : rec.category}</small>
        </div>
        {rec.rating && <div className="product-rating"><Stars rating={rec.rating} /> {rec.rating}</div>}
        {rec.festival_relevance && rec.festival_relevance !== 'None' && (
          <span className="festival-badge">🎉 {rec.festival_relevance}</span>
        )}
        <div className="explanation">💡 {rec.explanation}</div>

        <div className="signal-row">
          <span className="signal-label">Match confidence</span>
          <SignalBars score={score} />
        </div>

        <div className="feedback-row">
          <button className={`fb-btn ${fb === 'like' ? 'liked' : ''}`} onClick={() => handleFb('like')}>👍</button>
          <button className={`fb-btn ${fb === 'dislike' ? 'disliked' : ''}`} onClick={() => handleFb('dislike')}>👎</button>
          <button className="fb-btn" onClick={() => handleFb('not_interested')}>🚫</button>
        </div>
      </div>
    </div>
  )
}

export default function Recommendations() {
  const [userId,   setUserId]   = useState('')
  const [model,    setModel]    = useState('hybrid')
  const [profile,  setProfile]  = useState(null)
  const [recs,     setRecs]     = useState([])
  const [loading,  setLoading]  = useState(false)
  const [error,    setError]    = useState('')

  const fetchRecs = async () => {
    const id = parseInt(userId)
    if (!id || id < 1 || id > 10000) { setError('Enter a User ID between 1 and 10,000'); return }
    setError(''); setLoading(true)
    try {
      const [profileData, recData] = await Promise.all([api.user(id), api.recommend(id, model, 10)])
      setProfile(profileData)
      setRecs((recData.recommendations || []).map(r => ({ ...r, _userId: id })))
    } catch {
      setError('Could not fetch data. Is the Flask API running?')
    }
    setLoading(false)
  }

  const handleKey = (e) => { if (e.key === 'Enter') fetchRecs() }
  const tryRandom = () => setUserId(String(Math.floor(Math.random() * 10000) + 1))

  const MODELS = [
    { id:'hybrid',        label:'🔀 Hybrid',        sub:'CF + Content (best)' },
    { id:'collaborative', label:'👥 Collaborative', sub:'SVD matrix factorization' },
    { id:'content',       label:'📄 Content-Based', sub:'Feature similarity' },
  ]

  return (
    <div>
      <div className="page-header">
        <div className="page-title">Recommendations</div>
        <div className="page-subtitle">Enter a User ID to see personalized ML-powered recommendations</div>
      </div>

      <div className="rec-layout">
        <div className="user-panel">
          <div className="panel-card">
            <div className="panel-title">User ID</div>
            <div className="search-row">
              <input className="search-input" type="number" min="1" max="10000"
                     placeholder="e.g. 1234" value={userId}
                     onChange={e => setUserId(e.target.value)} onKeyDown={handleKey} />
              <button className="btn btn-orange" onClick={fetchRecs}>Go</button>
            </div>
            <button className="btn btn-ghost" style={{ marginTop:8, width:'100%' }} onClick={tryRandom}>
              🎲 Random user
            </button>
            {error && <div style={{ marginTop:8, fontSize:12, color:'var(--red)' }}>{error}</div>}
          </div>

          <div className="panel-card">
            <div className="panel-title">Model</div>
            {MODELS.map(m => (
              <div key={m.id} onClick={() => setModel(m.id)}
                   className={`model-option ${model === m.id ? 'active' : ''}`}>
                <div className="model-option-name">{m.label}</div>
                <div className="model-option-sub">{m.sub}</div>
              </div>
            ))}
          </div>

          {profile && (
            <div className="panel-card">
              <div className="profile-avatar">{profile.profile?.gender === 'Female' ? '👩' : '👨'}</div>
              <div className="profile-name">User #{profile.user_id}</div>
              <div className="profile-city">📍 {profile.profile?.city}</div>
              <div className="profile-divider" />

              <div className="profile-row"><span className="profile-key">Age</span><span className="profile-val">{profile.profile?.age} yrs</span></div>
              <div className="profile-row"><span className="profile-key">Gender</span><span className="profile-val">{profile.profile?.gender}</span></div>
              <div className="profile-row"><span className="profile-key">Income</span><span className="profile-val">{profile.profile?.income_level}</span></div>
              <div className="profile-row"><span className="profile-key">Device</span><span className="profile-val">{profile.profile?.device_type}</span></div>
              <div className="profile-row"><span className="profile-key">Purchases</span><span className="profile-val">{profile.profile?.frequency ?? 0}</span></div>
              <div className="profile-row">
                <span className="profile-key">Segment</span>
                <span className={`segment-badge seg-${profile.profile?.segment}`}>{profile.profile?.segment || '–'}</span>
              </div>
              <div className="profile-row"><span className="profile-key">Fav category</span>
                <span className="profile-val" style={{ fontSize:12 }}>{profile.profile?.preferred_category}</span>
              </div>

              {profile.profile?.remittance_receiver && (
                <div style={{ background:'#EEF2FF', borderRadius:8, padding:'7px 9px',
                              fontSize:12, color:'#4338CA', fontWeight:600, marginTop:8 }}>
                  💸 Remittance receiver
                </div>
              )}

              {profile.top_categories?.length > 0 && (
                <div style={{ marginTop:14 }}>
                  <div className="panel-title" style={{ marginBottom:8 }}>Top Categories</div>
                  <div style={{ display:'flex', gap:6, flexWrap:'wrap' }}>
                    {profile.top_categories.map(c => (
                      <span key={c} className="tag">{CAT_ICONS[c] || '📦'} {c}</span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        <div>
          {loading && <div className="loading"><div className="spinner" /><p>Generating recommendations…</p></div>}

          {!loading && recs.length === 0 && !profile && (
            <div className="empty">
              <div className="empty-icon">🎯</div>
              <div className="empty-text">Enter a User ID on the left to see recommendations</div>
            </div>
          )}

          {!loading && recs.length > 0 && (
            <>
              <div style={{ marginBottom:16, fontSize:14, fontWeight:700 }}>
                Top {recs.length} recommendations
                <span style={{ fontSize:12, color:'var(--muted)', marginLeft:8, fontWeight:500 }}>via {model} model</span>
              </div>
              <div className="rec-grid">
                {recs.map(rec => <ProductCard key={rec.product_id} rec={rec} />)}
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  )
}