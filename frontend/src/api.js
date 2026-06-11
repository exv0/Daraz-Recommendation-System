const BASE = 'http://localhost:5000/api'

const get = (url) => fetch(BASE + url).then(r => r.json())

export const api = {
  health:       ()         => get('/health'),
  stats:        ()         => get('/stats'),
  categories:   ()         => get('/categories'),
  users:        (page=1)   => get(`/users?page=${page}&limit=20`),
  user:         (id)       => get(`/user/${id}`),
  products:     (cat='', sort='popularity_count') =>
                             get(`/products?category=${cat}&sort=${sort}&limit=50`),
  product:      (id)       => get(`/product/${id}`),
  recommend:    (id, model='hybrid', n=10) =>
                             get(`/recommend/${id}?model=${model}&n=${n}`),
  similar:      (id, n=6)  => get(`/similar/${id}?n=${n}`),
  feedback:     (body)     => fetch(BASE + '/feedback', {
                               method: 'POST',
                               headers: { 'Content-Type': 'application/json' },
                               body: JSON.stringify(body)
                             }).then(r => r.json()),
}