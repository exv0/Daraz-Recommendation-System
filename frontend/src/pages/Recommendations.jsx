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

const catClass = (cat) =>
  'cat-' + (cat || '').replace(/[^a-zA-Z0-9]/g, (m) => '\\' + m)

function Stars({ rating }) {
  const full = Math.floor(rating)
  const half = rating % 1 >= 0.5
  return (
    <span className="stars">
      {'★'.repeat(full)}{'☆'.repeat(5 - full - (half ? 1 : 0))}{half ? '½' : ''}
    </span>
  )
}

function ProductCard({ rec, onFeedback }) {
  const [fb, setFb] = useState(null)

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
        {rec.rating && (
          <div className="product-rating">
            <Stars rating={rec.rating} /> {rec.rating}
          </div>
        )}

        {rec.festival_relevance && rec.festival_relevance !== 'None' && (
          <span className="festival-badge">🎉 {rec.festival_relevance}</span>
        )}

        <div className="explanation">💡 {rec.explanation}</div>

        <div className="score-bar-wrap">
          <div className="score-bar-label">
            <span>Match score</span>
            <span>{((rec.score || rec.hybrid_score || 0) * 100).toFixed(0)}%</span>
          </div>
          <div className="score-bar">
            <div className="score-fill"
                 style={{ width: `${(rec.score || rec.hybrid_score || 0) * 100}%` }} />
          </div>
        </div>

        <div className="feedback-row">
          <button className={`fb-btn ${fb === 'like' ? 'liked' : ''}`}
                  onClick={() => handleFb('like')}>👍</button>
          <button className={`fb-btn ${fb === 'dislike' ? 'disliked' : ''}`}
                  onClick={() => handleFb('dislike')}>👎</button>
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
    if (!id || id < 1 || id > 10000) {
      setError('Enter a User ID between 1 and 10,000')
      return
    }
    setError('')
    setLoading(true)
    try {
      const [profileData, recData] = await Promise.all([
        api.user(id),
        api.recommend(id, model, 10),
      ])
      setProfile(profileData)
      const tagged = (recData.recommendations || []).map(r => ({ ...r, _userId: id }))
      setRecs(tagged)
    } catch (e) {
      setError('Could not fetch data. Is the Flask API running?')
    }
    setLoading(false)
  }

  const handleKey = (e) => { if (e.key === 'Enter') fetchRecs() }

  const tryRandom = () => {
    const id = Math.floor(Math.random() * 10000) + 1
    setUserId(String(id))
  }

  return (
    <div>
      <div className="page-header">
        <div className="page-title">Recommendations</div>
        <div className="page-subtitle">
          Enter a User ID to see personalized ML-powered recommendations
        </div>
      </div>

      <div className="rec-layout">
        {/* Left panel */}
        <div className="user-panel">
          {/* Search */}
          <div className="user-search">
            <div className="search-label">User ID</div>
            <div className="search-row">
              <input className="search-input" type="number" min="1" max="10000"
                     placeholder="e.g. 1234" value={userId}
                     onChange={e => setUserId(e.target.value)}
                     onKeyDown={handleKey} />
              <button className="btn btn-orange" onClick={fetchRecs}>Go</button>
            </div>
            <button className="btn btn-ghost"
                    style={{ marginTop:8, width:'100%' }}
                    onClick={tryRandom}>🎲 Random user</button>
            {error && (
              <div style={{ marginTop:8, fontSize:12, color:'#E53E3E' }}>{error}</div>
            )}
          </div>

          {/* Model selector */}
          <div className="user-search">
            <div className="search-label">Model</div>
            {[
              { id:'hybrid',        label:'🔀 Hybrid',        sub:'CF + Content (best)' },
              { id:'collaborative', label:'👥 Collaborative', sub:'SVD matrix factorization' },
              { id:'content',       label:'📄 Content-Based', sub:'Feature similarity' },
            ].map(m => (
              <div key={m.id}
                   onClick={() => setModel(m.id)}
                   style={{
                     padding:'10px 12px', borderRadius:8, cursor:'pointer', marginBottom:6,
                     border:`1.5px solid ${model===m.id ? 'var(--orange)' : 'var(--border)'}`,
                     background: model===m.id ? 'var(--orange-lt)' : 'var(--bg)',
                   }}>
                <div style={{ fontSize:13, fontWeight:600,
                              color: model===m.id ? 'var(--orange)' : 'var(--text)' }}>
                  {m.label}
                </div>
                <div style={{ fontSize:11, color:'var(--muted)', marginTop:2 }}>{m.sub}</div>
              </div>
            ))}
          </div>

          {/* User profile */}
          {profile && (
            <div className="profile-card">
              <div className="profile-avatar">
                {profile.profile?.gender === 'Female' ? '👩' : '👨'}
              </div>
              <div className="profile-name">User #{profile.user_id}</div>
              <div className="profile-city">📍 {profile.profile?.city}</div>

              <div className="profile-rows">
                <div className="profile-row">
                  <span className="profile-key">Age</span>
                  <span className="profile-val">{profile.profile?.age} yrs</span>
                </div>
                <div className="profile-row">
                  <span className="profile-key">Gender</span>
                  <span className="profile-val">{profile.profile?.gender}</span>
                </div>
                <div className="profile-row">
                  <span className="profile-key">Income</span>
                  <span className="profile-val">{profile.profile?.income_level}</span>
                </div>
                <div className="profile-row">
                  <span className="profile-key">Device</span>
                  <span className="profile-val">{profile.profile?.device_type}</span>
                </div>
                <div className="profile-row">
                  <span className="profile-key">Purchases</span>
                  <span className="profile-val">{profile.profile?.frequency ?? 0}</span>
                </div>
                <div className="profile-row">
                  <span className="profile-key">Segment</span>
                  <span className={`segment-badge seg-${profile.profile?.segment}`}>
                    {profile.profile?.segment || '–'}
                  </span>
                </div>
                <div className="profile-row">
                  <span className="profile-key">Fav category</span>
                  <span className="profile-val" style={{ fontSize:12 }}>
                    {profile.profile?.preferred_category}
                  </span>
                </div>
                {profile.profile?.remittance_receiver && (
                  <div style={{ background:'#EEF2FF', borderRadius:6, padding:'6px 8px',
                                fontSize:12, color:'#4338CA', fontWeight:600 }}>
                    💸 Remittance receiver
                  </div>
                )}
              </div>

              {/* Top categories */}
              {profile.top_categories?.length > 0 && (
                <div style={{ marginTop:12 }}>
                  <div className="search-label" style={{ marginBottom:6 }}>
                    Top Categories
                  </div>
                  <div style={{ display:'flex', gap:6, flexWrap:'wrap' }}>
                    {profile.top_categories.map(c => (
                      <span key={c} className="tag">
                        {CAT_ICONS[c] || '📦'} {c}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Right panel — recommendations */}
        <div>
          {loading && (
            <div className="loading"><div className="spinner" /><p>Generating recommendations…</p></div>
          )}

          {!loading && recs.length === 0 && !profile && (
            <div className="empty">
              <div className="empty-icon">🎯</div>
              <div className="empty-text">
                Enter a User ID on the left to see recommendations
              </div>
            </div>
          )}

          {!loading && recs.length > 0 && (
            <>
              <div style={{ display:'flex', justifyContent:'space-between',
                            alignItems:'center', marginBottom:16 }}>
                <div style={{ fontSize:14, fontWeight:700 }}>
                  Top {recs.length} recommendations
                  <span style={{ fontSize:12, color:'var(--muted)', marginLeft:8 }}>
                    via {model} model
                  </span>
                </div>
              </div>
              <div className="rec-grid">
                {recs.map(rec => (
                  <ProductCard key={rec.product_id} rec={rec} />
                ))}
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  )
}