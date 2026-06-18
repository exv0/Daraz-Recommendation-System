import { useEffect, useState } from 'react'
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
  return <span className="stars">{'★'.repeat(Math.round(rating))}{'☆'.repeat(5-Math.round(rating))}</span>
}

export default function Products() {
  const [products,  setProducts]  = useState([])
  const [cats,      setCats]      = useState([])
  const [selCat,    setSelCat]    = useState('')
  const [sort,      setSort]      = useState('popularity_count')
  const [search,    setSearch]    = useState('')
  const [loading,   setLoading]   = useState(true)

  useEffect(() => { api.categories().then(d => setCats(d.categories || [])) }, [])

  useEffect(() => {
    setLoading(true)
    api.products(selCat, sort)
      .then(d => setProducts(d.products || []))
      .finally(() => setLoading(false))
  }, [selCat, sort])

  const filtered = products.filter(p =>
    !search || p.product_name?.toLowerCase().includes(search.toLowerCase())
      || p.brand?.toLowerCase().includes(search.toLowerCase())
  )

  return (
    <div>
      <div className="page-header">
        <div className="page-title">Products</div>
        <div className="page-subtitle">Browse the product catalog</div>
      </div>

      <div className="filter-bar">
        <button className={`cat-filter ${selCat==='' ? 'active' : ''}`} onClick={() => setSelCat('')}>All</button>
        {cats.map(c => (
          <button key={c.name} className={`cat-filter ${selCat===c.name ? 'active' : ''}`}
                  onClick={() => setSelCat(c.name)}>
            {CAT_ICONS[c.name]} {c.name}
          </button>
        ))}
      </div>

      <div style={{ display:'flex', gap:12, marginBottom:20, alignItems:'center' }}>
        <input className="search-input" style={{ maxWidth:280 }} placeholder="🔍 Search products…"
               value={search} onChange={e => setSearch(e.target.value)} />
        <select value={sort} onChange={e => setSort(e.target.value)}
                style={{ padding:'9px 12px', borderRadius:6, border:'1.5px solid var(--border)',
                         fontSize:13, fontFamily:'inherit', background:'var(--card)',
                         cursor:'pointer', outline:'none', color:'var(--text)' }}>
          <option value="popularity_count">Sort: Popularity</option>
          <option value="rating">Sort: Rating</option>
          <option value="price_npr">Sort: Price ↑</option>
          <option value="conversion_rate">Sort: Conversion</option>
        </select>
        <span style={{ fontSize:13, color:'var(--muted)' }}>{filtered.length} products</span>
      </div>

      {loading ? (
        <div className="loading"><div className="spinner" /><p>Loading products…</p></div>
      ) : (
        <div className="products-grid">
          {filtered.map(p => (
            <div key={p.product_id} className="product-card">
              <div className={`product-img cat-${CSS.escape(p.category || '')}`} style={{ height:100 }}>
                {CAT_ICONS[p.category] || '📦'}
              </div>
              <div className="product-body">
                <div className="product-name">{p.product_name}</div>
                <div className="product-brand">{p.brand}</div>
                <div className="product-price">NPR {p.price_npr?.toLocaleString()}</div>
                <div className="product-rating">
                  <Stars rating={p.rating} /> {p.rating}
                  <span style={{ marginLeft:4 }}>({p.review_count})</span>
                </div>
                <div style={{ display:'flex', gap:6, marginTop:8, flexWrap:'wrap' }}>
                  <span className="tag">👁 {p.popularity_count} views</span>
                  {p.festival_relevance && p.festival_relevance !== 'None' && (
                    <span className="festival-badge">🎉 {p.festival_relevance}</span>
                  )}
                </div>
                <div style={{ marginTop:10 }}>
                  <div className="signal-row" style={{ marginTop:0 }}>
                    <span className="signal-label">Conversion</span>
                    <span className="signal-pct">{(p.conversion_rate * 100).toFixed(1)}%</span>
                  </div>
                  <div className="funnel-track" style={{ marginTop:5 }}>
                    <div className="funnel-fill"
                         style={{ width:`${Math.min(p.conversion_rate*100*5, 100)}%`, background:'var(--green)' }} />
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}