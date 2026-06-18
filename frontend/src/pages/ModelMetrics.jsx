import { useEffect, useState } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  Legend, Cell
} from 'recharts'
import { api } from '../api.js'

const MODEL_COLORS = {
  'Collaborative (SVD)': '#2563EB',
  'Content-Based':       '#059669',
  'Hybrid (CF + CB)':    '#F85606',
  'Neural CF (NCF)':     '#7C3AED',
}

const METRIC_INFO = {
  precision: { label: 'Precision@10', desc: 'Of 10 recommendations, how many were relevant?' },
  recall:    { label: 'Recall@10',    desc: 'Of all relevant items, what fraction was found?' },
  f1:        { label: 'F1@10',        desc: 'Harmonic mean of Precision and Recall' },
  ndcg:      { label: 'NDCG@10',      desc: 'Ranking quality — rewards relevant items ranked higher' },
}

function MetricCard({ label, desc, models, metric }) {
  const best = Object.entries(models).reduce((a, [k, v]) => v[metric] > a[1] ? [k, v[metric]] : a, ['', 0])
  return (
    <div className="card card-pad" style={{ flex: 1 }}>
      <div style={{ fontSize: 13, fontWeight: 700 }}>{label}</div>
      <div style={{ fontSize: 11, color: 'var(--muted)', marginBottom: 14 }}>{desc}</div>
      {Object.entries(models).map(([name, vals]) => (
        <div key={name} style={{ marginBottom: 9 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, marginBottom: 4 }}>
            <span style={{ fontWeight: name === best[0] ? 700 : 400, color: MODEL_COLORS[name] }}>{name}</span>
            <span style={{ fontWeight: 700 }}>{(vals[metric] * 100).toFixed(3)}%</span>
          </div>
          <div className="funnel-track">
            <div className="funnel-fill" style={{ width: `${(vals[metric] / best[1]) * 100}%`, background: MODEL_COLORS[name] }} />
          </div>
        </div>
      ))}
    </div>
  )
}

export default function ModelMetrics() {
  const [metrics, setMetrics] = useState(null)
  const [shap,    setShap]    = useState(null)
  const [sig,     setSig]     = useState(null)
  const [loading, setLoading] = useState(true)
  const [err,     setErr]     = useState('')

  useEffect(() => {
    Promise.all([api.metrics(), api.shap(), api.significance().catch(() => null)])
      .then(([m, s, sg]) => { setMetrics(m); setShap(s); setSig(sg) })
      .catch(() => setErr('Could not load metrics. Is the Flask API running?'))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div className="loading"><div className="spinner" /><p>Loading metrics…</p></div>
  if (err) return <div className="empty"><div className="empty-icon">⚠️</div><div className="empty-text">{err}</div></div>

  const models     = metrics?.models || {}
  const modelNames = Object.keys(models)

  const barData = ['precision','recall','f1','ndcg'].map(metric => ({
    metric: METRIC_INFO[metric]?.label || metric,
    ...Object.fromEntries(modelNames.map(n => [n, +(models[n][metric] * 100).toFixed(4)])),
  }))

  const radarData = ['precision','recall','f1','ndcg'].map(m => ({
    metric: METRIC_INFO[m]?.label || m,
    ...Object.fromEntries(modelNames.map(n => [n, +(models[n][m] * 1000).toFixed(4)])),
  }))

  const shapTop = shap
    ? shap.feature_labels.slice(0, 10).map((label, i) => ({ label, value: +(shap.shap_values[i] * 100).toFixed(4) }))
    : []

  return (
    <div>
      <div className="page-header">
        <div className="page-title">Model Evaluation</div>
        <div className="page-subtitle">
          Precision@10, Recall@10, F1@10, NDCG@10 — evaluated on {metrics?.n_eval_users} held-out users
        </div>
      </div>

      <div style={{ background: 'linear-gradient(135deg,#F85606,#FF8C42)', borderRadius: 14,
                    padding: '18px 26px', marginBottom: 24, color: '#fff',
                    display: 'flex', alignItems: 'center', gap: 18, boxShadow:'var(--shadow-orange)' }}>
        <span style={{ fontSize: 34 }}>🏆</span>
        <div>
          <div style={{ fontSize: 12.5, opacity: 0.85 }}>Best performing model (NDCG@10)</div>
          <div style={{ fontSize: 22, fontWeight: 800 }}>{metrics?.best_model}</div>
        </div>
        <div style={{ marginLeft: 'auto', textAlign: 'right' }}>
          <div style={{ fontSize: 12, opacity: 0.85 }}>Evaluation K</div>
          <div style={{ fontSize: 24, fontWeight: 800 }}>K = {metrics?.k}</div>
        </div>
      </div>

      <div style={{ display: 'flex', gap: 16, marginBottom: 24 }}>
        {Object.entries(METRIC_INFO).map(([key, info]) => (
          <MetricCard key={key} label={info.label} desc={info.desc} models={models} metric={key} />
        ))}
      </div>

      <div className="chart-card" style={{ marginBottom: 24 }}>
        <div className="chart-title">Model Comparison — All Metrics (×100 for readability)</div>
        <ResponsiveContainer width="100%" height={260}>
          <BarChart data={barData} barGap={4}>
            <XAxis dataKey="metric" tick={{ fontSize: 12 }} axisLine={false} tickLine={false} />
            <YAxis tick={{ fontSize: 11 }} axisLine={false} tickLine={false} tickFormatter={v => v.toFixed(3)} />
            <Tooltip formatter={(v) => `${v.toFixed(4)}%`} />
            <Legend iconType="circle" iconSize={10} />
            {modelNames.map(name => (
              <Bar key={name} dataKey={name} fill={MODEL_COLORS[name]} radius={[4,4,0,0]} />
            ))}
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 24 }}>
        <div className="chart-card">
          <div className="chart-title">Model Radar (scores ×1000)</div>
          <ResponsiveContainer width="100%" height={260}>
            <RadarChart data={radarData}>
              <PolarGrid />
              <PolarAngleAxis dataKey="metric" tick={{ fontSize: 11 }} />
              <PolarRadiusAxis tick={{ fontSize: 9 }} />
              {modelNames.map(name => (
                <Radar key={name} name={name} dataKey={name} stroke={MODEL_COLORS[name]}
                       fill={MODEL_COLORS[name]} fillOpacity={0.15} />
              ))}
              <Legend iconType="circle" iconSize={10} />
            </RadarChart>
          </ResponsiveContainer>
        </div>

        <div className="chart-card">
          <div className="chart-title">
            SHAP Feature Importance
            <span className="chart-badge">acc: {(shap?.classifier_accuracy * 100).toFixed(1)}%</span>
          </div>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={shapTop} layout="vertical" barSize={14}>
              <XAxis type="number" tick={{ fontSize: 10 }} axisLine={false} tickLine={false} tickFormatter={v => v.toFixed(1)} />
              <YAxis type="category" dataKey="label" tick={{ fontSize: 11 }} width={140} axisLine={false} tickLine={false} />
              <Tooltip formatter={(v) => [`${v.toFixed(4)}`, 'Mean |SHAP|']} />
              <Bar dataKey="value" radius={[0,4,4,0]}>
                {shapTop.map((_, i) => <Cell key={i} fill={`hsl(${20 + i * 16}, 85%, ${52 - i * 2}%)`} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="chart-card">
        <div className="chart-title">Raw Metrics Table</div>
        <table className="rfm-table">
          <thead>
            <tr><th>Model</th><th>Precision@10</th><th>Recall@10</th><th>F1@10</th><th>NDCG@10</th></tr>
          </thead>
          <tbody>
            {modelNames.map(name => {
              const m = models[name]
              const isBest = name === metrics?.best_model
              return (
                <tr key={name}>
                  <td><span style={{ fontWeight: 700, color: MODEL_COLORS[name] }}>{isBest ? '🏆 ' : ''}{name}</span></td>
                  {['precision','recall','f1','ndcg'].map(k => (
                    <td key={k} style={{ fontWeight: isBest ? 700 : 400 }}>{(m[k] * 100).toFixed(4)}%</td>
                  ))}
                </tr>
              )
            })}
          </tbody>
        </table>
        <div style={{ marginTop: 14, padding: '11px 14px', background: 'var(--bg)', borderRadius: 8, fontSize: 12, color: 'var(--muted)' }}>
          ℹ️ Low absolute values are expected on sparse synthetic datasets (95.7% sparsity).
          The metric that matters for the thesis is the <strong>relative ranking</strong> —
          hybrid outperforms single-method baselines, validating the research hypothesis.
        </div>
      </div>

      {/* Statistical Significance Testing */}
      {sig && (
        <div className="chart-card" style={{ marginTop: 24 }}>
          <div className="chart-title">
            🔬 Statistical Significance Testing
            <span className="chart-badge">Wilcoxon + paired t-test, α={sig.alpha}</span>
          </div>
          <p style={{ fontSize: 12.5, color: 'var(--muted)', marginBottom: 16, lineHeight: 1.6 }}>
            Tests whether the Hybrid model's NDCG@10 improvement over each baseline is statistically
            significant, not just due to chance. Evaluated on {sig.n_eval_users} held-out users.
          </p>

          <table className="rfm-table">
            <thead>
              <tr>
                <th>Comparison</th>
                <th>Wilcoxon p</th>
                <th>t-test p</th>
                <th>Cohen's d</th>
                <th>Effect size</th>
                <th>Significant?</th>
                <th>Improvement</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(sig.comparisons || {}).map(([key, c]) => (
                <tr key={key}>
                  <td style={{ fontWeight: 700 }}>{c.comparison}</td>
                  <td>{c.wilcoxon_p !== null ? c.wilcoxon_p.toFixed(4) : '—'}</td>
                  <td>{c.t_p !== null ? c.t_p.toFixed(4) : '—'}</td>
                  <td>{c.cohens_d}</td>
                  <td>
                    <span className="tag" style={{
                      background: c.effect_size === 'large' ? '#FEF3C7'
                                : c.effect_size === 'medium' ? '#DBEAFE' : '#F3F4F6',
                      color: c.effect_size === 'large' ? '#92400E'
                           : c.effect_size === 'medium' ? '#1E40AF' : '#6B7280',
                      border: 'none',
                    }}>{c.effect_size}</span>
                  </td>
                  <td>
                    <span style={{
                      fontWeight: 700,
                      color: c.significant ? 'var(--green)' : 'var(--muted)',
                    }}>
                      {c.significant ? '✓ Yes' : '✗ No'}
                    </span>
                  </td>
                  <td style={{
                    fontWeight: 600,
                    color: c.improvement_pct > 0 ? 'var(--green)' : 'var(--red)',
                  }}>
                    {c.improvement_pct > 0 ? '+' : ''}{c.improvement_pct}%
                  </td>
                </tr>
              ))}
            </tbody>
          </table>

          <div style={{ marginTop: 14, padding: '11px 14px', background: '#EEF2FF', borderRadius: 8,
                        fontSize: 12, color: '#3730A3', lineHeight: 1.6 }}>
            📋 <strong>(thesis ma):</strong> "{sig.null_hypothesis}." was tested
            using {sig.test_methods?.join(' and ')}. Where p &lt; {sig.alpha}, the null hypothesis
            is rejected, confirming the observed improvement is statistically significant rather
            than due to random variation in the held-out sample.
          </div>
        </div>
      )}
    </div>
  )
}