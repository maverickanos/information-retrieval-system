const status = document.querySelector('#status');
const results = document.querySelector('#results');
const evaluation = document.querySelector('#evaluation');

async function request(path, options = {}) {
  let response;
  try {
    response = await fetch(path, { headers: { 'Content-Type': 'application/json' }, ...options });
  } catch (error) {
    throw new Error(`Could not reach ${path}: ${error.message}`);
  }

  const responseText = await response.text();
  if (!responseText.trim()) {
    throw new Error(`${path} returned an empty response (HTTP ${response.status}).`);
  }

  let body;
  try {
    body = JSON.parse(responseText);
  } catch {
    const isHtml = response.headers.get('content-type')?.includes('text/html') || /^\s*<!doctype html/i.test(responseText);
    const format = isHtml ? 'an HTML page' : 'a non-JSON response';
    throw new Error(`${path} returned ${format} (HTTP ${response.status}). Check that the request reaches the backend API.`);
  }

  if (!response.ok) {
    const detail = body && typeof body === 'object' && typeof body.detail === 'string' ? body.detail : 'Request failed';
    throw new Error(`${detail} (HTTP ${response.status})`);
  }
  return body;
}

function score(label, value) {
  return `<div class="score"><span class="metric-label">${label}</span><strong>${value === null || value === undefined ? '—' : Number(value).toFixed(6)}</strong></div>`;
}

function renderSearch(data) {
  results.innerHTML = data.results.map(item => `<article class="result"><div class="result-head"><h2>${item.title}</h2><span class="rank">FINAL RANK ${item.final_rank} · ${item.id}</span></div><p class="content">${item.snippet}</p><div class="scores">${score('BM25', item.bm25_score)}${score('TF-IDF', item.tfidf_score)}${score('Semantic', item.semantic_score)}${score('RRF Score', item.rrf_score)}${score('BM25 Rank', item.bm25_rank)}${score('TF-IDF Rank', item.tfidf_rank)}${score('Semantic Rank', item.semantic_rank)}</div></article>`).join('') || '<p class="empty">No documents matched.</p>';
}

document.querySelector('#search-button').addEventListener('click', async () => {
  try { status.textContent = 'Searching...'; renderSearch(await request('/search', { method: 'POST', body: JSON.stringify({ query: document.querySelector('#query').value, top_k: Number(document.querySelector('#top-k').value) }) })); status.textContent = 'Search complete'; }
  catch (error) { status.textContent = error.message; }
});
document.querySelector('#query').addEventListener('keydown', event => { if (event.key === 'Enter') document.querySelector('#search-button').click(); });
document.querySelector('#index-button').addEventListener('click', async () => { try { status.textContent = 'Indexing...'; const data = await request('/index', { method: 'POST' }); status.textContent = `${data.document_count} documents indexed`; } catch (error) { status.textContent = error.message; } });
document.querySelector('#evaluate-button').addEventListener('click', async () => { try { status.textContent = 'Evaluating...'; const data = await request('/evaluate', { method: 'POST', body: JSON.stringify({ k: Number(document.querySelector('#top-k').value) }) }); evaluation.hidden = false; evaluation.innerHTML = `<h2>Evaluation at K=${data.k}</h2><div class="eval-grid"><span>Precision: <strong>${data.mean_precision_at_k.toFixed(4)}</strong></span><span>Recall: <strong>${data.mean_recall_at_k.toFixed(4)}</strong></span><span>MRR: <strong>${data.mrr.toFixed(4)}</strong></span></div>`; status.textContent = 'Evaluation complete'; } catch (error) { status.textContent = error.message; } });