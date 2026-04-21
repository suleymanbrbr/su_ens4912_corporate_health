import React, { useState, useEffect, useRef, useCallback } from 'react';
import ForceGraph2D from 'react-force-graph-2d';
import { Search, X, RefreshCw, Info, GitBranch, Loader, Share2 } from 'lucide-react';

const API = (path) => fetch(path, {
  headers: { 'Authorization': `Bearer ${localStorage.getItem('token')}` }
});

// Turkish labels for node types
const NODE_TYPE_TR = {
  RULE:       'SUT Kuralı',
  DRUG:       'İlaç / Etken Madde',
  DIAGNOSIS:  'Teşhis / Hastalık',
  SPECIALIST: 'Uzman / Hekim',
  CONDITION:  'Koşul / Şart',
  DOCUMENT:   'Belge / Rapor',
  DEVICE:     'Tıbbi Cihaz',
  DOSAGE:     'Doz Bilgisi',
  AGE_LIMIT:  'Yaş Sınırı',
  EXCLUSION:  'Dışlama / Kontrendikasyon',
};

// Turkish labels for edge relations
const RELATION_TR = {
  COVERS:              'Kapsar',
  TREATS:              'Tedavi Eder',
  ISSUED_BY:           'Düzenlenir (Kurum)',
  PRESCRIBED_BY:       'Reçete Edilir (Hekim)',
  REQUIRES_CONDITION:  'Koşul Gerektirir',
  MUST_FAIL_FIRST:     'Önce Başarısız Olmalı',
  HAS_LIMIT:           'Sınırlama Var',
  NOT_COVERED_FOR:     'Kapsam Dışı',
  HAS_SUBRULE:         'Alt Kural İçerir',
  HAS_DOSAGE:          'Doz Bilgisi',
  CONTRAINDICATED_FOR: 'Kontrendike',
  HAS_AGE_LIMIT:       'Yaş Sınırı',
  APPROVED_BY:         'Onaylanır (Birim)',
  REQUIRES_REPORT:     'Rapor Gerektirir',
  FUNDED_BY:           'Finanse Edilir',
};

const NODE_COLORS = {
  RULE:       '#3b82f6',
  DRUG:       '#10b981',
  DIAGNOSIS:  '#ef4444',
  SPECIALIST: '#f59e0b',
  CONDITION:  '#8b5cf6',
  DOCUMENT:   '#06b6d4',
  DEVICE:     '#ec4899',
  DOSAGE:     '#84cc16',
  AGE_LIMIT:  '#f97316',
  EXCLUSION:  '#6b7280',
};

function Badge({ type }) {
  return (
    <span style={{
      display: 'inline-block', padding: '0.2rem 0.6rem', borderRadius: '12px',
      fontSize: '0.7rem', fontWeight: 700, letterSpacing: '0.5px',
      background: NODE_COLORS[type] + '22', color: NODE_COLORS[type],
      border: `1px solid ${NODE_COLORS[type]}44`,
    }}>
      {NODE_TYPE_TR[type] || type}
    </span>
  );
}

function KnowledgeGraph() {
  const [graphData, setGraphData] = useState({ nodes: [], links: [] });
  const [loading, setLoading] = useState(true);
  const [rebuilding, setRebuilding] = useState(false);
  const [stats, setStats]   = useState(null);
  const [search, setSearch] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [searching, setSearching]         = useState(false);
  const [selectedNodes, setSelectedNodes] = useState([]); // Support for multiple selection
  const [neighborData, setNeighborData]   = useState([]); // Array of neighbor results
  const [pathMode, setPathMode]           = useState(false);
  const [pathFrom, setPathFrom]           = useState(null);
  const [pathTo, setPathTo]               = useState(null);
  const [pathResult, setPathResult]       = useState(null);
  const [pathLoading, setPathLoading]     = useState(false);
  const [dimensions, setDimensions]       = useState({ width: 900, height: 600 });
  const [filterType, setFilterType]       = useState('');
  const [error, setError] = useState('');

  const fgRef  = useRef();
  const contRef = useRef();
  const searchTimer = useRef();

  // ─── Resize observer ────────────────────────────────────────────────────────
  useEffect(() => {
    const update = () => {
      if (contRef.current) {
        setDimensions({ width: contRef.current.clientWidth, height: contRef.current.clientHeight });
      }
    };
    update();
    window.addEventListener('resize', update);
    return () => window.removeEventListener('resize', update);
  }, []);

  // ─── Load stats + initial graph ─────────────────────────────────────────────
  useEffect(() => {
    loadStats();
    loadGraph();
  }, [filterType]);

  const loadStats = async () => {
    try {
      const r = await API('/api/kg/stats');
      if (r.ok) setStats(await r.json());
    } catch {}
  };

  const loadGraph = async () => {
    setLoading(true);
    setError('');
    try {
      const url = filterType
        ? `/api/kg/nodes?limit=200&type=${filterType}`
        : '/api/kg/nodes?limit=200';
      const r = await API(url);
      if (!r.ok) throw new Error('Grafik verisi alınamadı.');
      const data = await r.json();

      // Fetch edges for visible nodes
      const nodeIds = new Set(data.nodes.map(n => n.node_id));
      // Load graph with visible nodes; edges from neighbor calls
      setGraphData({
        nodes: data.nodes.map(n => ({ ...n, id: n.node_id })),
        links: []
      });
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  // ─── Search ─────────────────────────────────────────────────────────────────
  const handleSearch = (val) => {
    setSearch(val);
    clearTimeout(searchTimer.current);
    if (!val.trim()) { setSearchResults([]); return; }
    searchTimer.current = setTimeout(async () => {
      setSearching(true);
      try {
        const r = await API(`/api/kg/nodes?q=${encodeURIComponent(val)}&limit=10`);
        if (r.ok) {
          const data = await r.json();
          setSearchResults(data.nodes || []);
        }
      } catch {}
      setSearching(false);
    }, 350);
  };

  // ─── Select node(s) ──────────────────────────────────────────
  const selectNodes = useCallback(async (nodesToSelect, append = false) => {
    const updatedList = append ? [...selectedNodes, ...nodesToSelect] : nodesToSelect;
    // Deduplicate by ID
    const uniqueList = Array.from(new Map(updatedList.map(n => [n.node_id || n.id, n])).values());
    setSelectedNodes(uniqueList);
    setSearchResults([]);
    setSearch('');
    
    try {
      const allNeighbors = await Promise.all(uniqueList.map(async (node) => {
        const r = await API(`/api/kg/node/${encodeURIComponent(node.node_id || node.id)}`);
        return r.ok ? await r.json() : null;
      }));
      
      const filteredResults = allNeighbors.filter(Boolean);
      setNeighborData(filteredResults);

      // Merge new nodes/links into graph data for visualization
      setGraphData(prev => {
        const existingNodeIds = new Set(prev.nodes.map(n => n.id));
        const existingLinkIds = new Set(prev.links.map(l => l.id));
        const mergedNodes = [...prev.nodes];
        const mergedLinks = [...prev.links];

        filteredResults.forEach(data => {
          const fetchedNodes = [data.node, ...(data.neighbors || []).map(nb => nb.node)].filter(Boolean);
          const fetchedLinks = (data.neighbors || []).map(nb => ({
            source: nb.edge.source_id,
            target: nb.edge.target_id,
            relation: nb.edge.relation,
            id: nb.edge.edge_id,
          }));

          fetchedNodes.forEach(n => {
            if (!existingNodeIds.has(n.node_id)) {
              mergedNodes.push({ ...n, id: n.node_id });
              existingNodeIds.add(n.node_id);
            }
          });
          fetchedLinks.forEach(l => {
            if (!existingLinkIds.has(l.id)) {
              mergedLinks.push(l);
              existingLinkIds.add(l.id);
            }
          });
        });
        return { nodes: mergedNodes, links: mergedLinks };
      });

      // Focus on the first/primary node selected
      if (uniqueList.length > 0) {
        const primary = uniqueList[uniqueList.length - 1];
        setTimeout(() => {
          const n = fgRef.current?.getGraphData().nodes.find(node_item => node_item.id === (primary.node_id || primary.id));
          if (fgRef.current && n) {
            fgRef.current.centerAt(n.x, n.y, 800);
            fgRef.current.zoom(2.5, 800);
          }
          if (!append) {
            const el = document.getElementById('relationships-header');
            if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
          }
        }, 450);
      }
    } catch (err) {
      console.error('Multi-node selection error:', err);
    }
  }, [selectedNodes]);

  // ─── Path finder ────────────────────────────────────────────────────────────
  const findPath = async () => {
    if (!pathFrom || !pathTo) return;
    setPathLoading(true);
    setPathResult(null);
    try {
      const r = await API(`/api/kg/path?from_id=${encodeURIComponent(pathFrom.node_id)}&to_id=${encodeURIComponent(pathTo.node_id)}&max_hops=3`);
      if (r.ok) setPathResult(await r.json());
    } catch {}
    setPathLoading(false);
  };

  // ─── Rebuild KG ─────────────────────────────────────────────────────────────
  const handleRebuild = async () => {
    if (!window.confirm('KG yeniden oluşturulacak. Bu işlem 5-10 dakika sürebilir. Devam et?')) return;
    setRebuilding(true);
    try {
      const r = await fetch('/api/admin/kg/rebuild', {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${localStorage.getItem('token')}` }
      });
      const d = await r.json();
      alert(d.message || 'Başlatıldı.');
    } catch { alert('Hata oluştu.'); }
    setRebuilding(false);
  };



  return (
    <div style={{ display: 'flex', height: '100%', position: 'relative', background: 'var(--bg)' }}>

      {/* ── Left Panel ─────────────────────────────────────── */}
      <div style={{ width: '340px', flexShrink: 0, borderRight: '1px solid var(--border)', display: 'flex', flexDirection: 'column', background: 'var(--card-bg)', zIndex: 10, height: '100%', overflow: 'hidden', position: 'relative' }}>

        {/* Header */}
        <div style={{ padding: '1.25rem', borderBottom: '1px solid var(--border)' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.75rem' }}>
            <h2 style={{ margin: 0, fontSize: '1rem', fontWeight: 700, display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
              <Share2 size={18} color="var(--primary)" /> Bilgi Grafiği
            </h2>
            <button onClick={handleRebuild} disabled={rebuilding} title="KG Yeniden Oluştur"
              style={{ background: 'transparent', border: '1px solid var(--border)', borderRadius: '8px', padding: '0.4rem', cursor: 'pointer', color: 'var(--text-muted)', display: 'flex' }}>
              <RefreshCw size={14} className={rebuilding ? 'spin' : ''} />
            </button>
          </div>

          {/* Stats */}
          {stats && (
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.5rem', marginBottom: '0.75rem' }}>
              <div style={{ padding: '0.5rem', background: '#e0f2fe', borderRadius: '8px', textAlign: 'center' }}>
                <div style={{ fontSize: '1.2rem', fontWeight: 700, color: '#0284c7' }}>{stats.node_count ?? '—'}</div>
                <div style={{ fontSize: '0.65rem', color: '#0284c7', textTransform: 'uppercase' }}>Düğüm</div>
              </div>
              <div style={{ padding: '0.5rem', background: '#dcfce7', borderRadius: '8px', textAlign: 'center' }}>
                <div style={{ fontSize: '1.2rem', fontWeight: 700, color: '#16a34a' }}>{stats.edge_count ?? '—'}</div>
                <div style={{ fontSize: '0.65rem', color: '#16a34a', textTransform: 'uppercase' }}>İlişki</div>
              </div>
            </div>
          )}

          {/* Search */}
          <div style={{ position: 'relative' }}>
            <Search size={14} style={{ position: 'absolute', left: 10, top: '50%', transform: 'translateY(-50%)', color: 'var(--text-muted)' }} />
            <input
              value={search}
              onChange={e => handleSearch(e.target.value)}
              placeholder="Düğüm ara (ilaç, teşhis...)"
              style={{ width: '100%', padding: '0.55rem 0.75rem 0.55rem 2rem', borderRadius: '8px', border: '1px solid var(--border)', fontSize: '0.85rem', boxSizing: 'border-box', background: 'var(--bg)' }}
            />
            {searching && <Loader size={12} style={{ position: 'absolute', right: 10, top: '50%', transform: 'translateY(-50%)', color: 'var(--text-muted)' }} />}
          </div>

          {/* Search results dropdown */}
          {searchResults.length > 0 && (
            <div style={{ marginTop: '0.5rem', background: 'var(--bg)', border: '1px solid var(--border)', borderRadius: '8px', maxHeight: '200px', overflowY: 'auto' }}>
              {searchResults.map(n => (
                <div key={n.node_id} onClick={() => selectNodes([n])}
                  style={{ padding: '0.6rem 0.75rem', cursor: 'pointer', borderBottom: '1px solid var(--border)', transition: 'background 0.15s' }}
                  onMouseEnter={e => e.currentTarget.style.background = '#f1f5f9'}
                  onMouseLeave={e => e.currentTarget.style.background = 'transparent'}>
                  <div style={{ fontSize: '0.8rem', fontWeight: 600 }}>{n.label}</div>
                  <Badge type={n.type} />
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Filters */}
        <div style={{ padding: '1rem', borderBottom: '1px solid var(--border)', background: 'rgba(0,0,0,0.02)' }}>
          <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', fontWeight: 700, textTransform: 'uppercase', marginBottom: '0.5rem' }}>Tür Filtresi</div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.4rem' }}>
            <button onClick={() => setFilterType('')} style={{ padding: '0.3rem 0.6rem', borderRadius: '20px', fontSize: '0.72rem', border: '1px solid var(--border)', background: filterType === '' ? 'var(--text-main)' : 'white', color: filterType === '' ? 'white' : 'var(--text-main)', cursor: 'pointer' }}>Tümü</button>
            {Object.entries(NODE_TYPE_TR).map(([key, label]) => (
              <button key={key} onClick={() => setFilterType(key)}
                style={{ padding: '0.3rem 0.6rem', borderRadius: '20px', fontSize: '0.72rem', border: `1px solid ${filterType === key ? NODE_COLORS[key] : 'var(--border)'}`, background: filterType === key ? NODE_COLORS[key] : 'white', color: filterType === key ? 'white' : 'var(--text-main)', cursor: 'pointer' }}>
                {label}
              </button>
            ))}
          </div>
        </div>

        {/* Node Detail Panel */}
        <div style={{ 
          flex: '1 1 0%', 
          overflowY: 'auto', 
          overflowX: 'hidden',
          padding: '1rem', 
          minHeight: 0, 
          scrollbarGutter: 'stable',
          display: 'flex',
          flexDirection: 'column',
          borderBottom: '1px solid var(--border)'
        }}>
          {selectedNodes.length === 0 && !pathMode && (
            <div style={{ color: 'var(--text-muted)', fontSize: '0.85rem', textAlign: 'center', marginTop: '2rem' }}>
              <Info size={28} style={{ opacity: 0.3, display: 'block', margin: '0 auto 0.5rem' }} />
              Düğüme tıklayın veya arama yapın
            </div>
          )}

          {selectedNodes.length > 0 && (
            <>
              <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', marginBottom: '1rem' }}>
                <h3 style={{ margin: 0, fontSize: '0.85rem', color: 'var(--text-muted)' }}>Seçili Düğümler ({selectedNodes.length})</h3>
                <button onClick={() => { setSelectedNodes([]); setNeighborData([]); }}
                  style={{ background: 'transparent', border: 'none', cursor: 'pointer', color: 'var(--text-muted)', fontSize: '0.75rem', textDecoration: 'underline' }}>
                  Temizle
                </button>
              </div>

              {selectedNodes.map((node, idx) => (
                <div key={node.node_id || idx} style={{ marginBottom: '1.5rem', borderBottom: idx < selectedNodes.length - 1 ? '1px dashed var(--border)' : 'none', paddingBottom: '0.5rem' }}>
                  <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
                    <div style={{ fontWeight: 700, fontSize: '0.9rem' }}>{node.label || node.node_id}</div>
                    <button onClick={() => selectNodes(selectedNodes.filter(n => (n.node_id || n.id) !== (node.node_id || node.id)))}
                      style={{ background: 'transparent', border: 'none', cursor: 'pointer', color: 'var(--text-muted)' }}>
                      <X size={14} />
                    </button>
                  </div>
                  <Badge type={node.type} />
                  
                  {node.text_content && (
                    <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', background: 'var(--bg)', padding: '0.5rem', borderRadius: '6px', marginTop: '0.5rem', maxHeight: '80px', overflowY: 'auto' }}>
                      {node.text_content}
                    </div>
                  )}
                </div>
              ))}

              <div id="relationships-header" style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontWeight: 700, textTransform: 'uppercase', marginBottom: '0.5rem', paddingTop: '0.5rem', borderTop: '1px solid var(--border)' }}>
                Tüm İlişkiler ({neighborData.reduce((acc, curr) => acc + (curr.neighbors?.length || 0), 0)})
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem' }}>
                {neighborData.flatMap((data, dIdx) => 
                  (data.neighbors || []).map((nb, i) => (
                    <div key={`${dIdx}-${i}`}
                      onClick={() => selectNodes([{ ...nb.node, node_id: nb.node.node_id }])}
                      style={{ padding: '0.5rem 0.75rem', background: '#f8fafc', borderRadius: '8px', cursor: 'pointer', borderLeft: `3px solid ${NODE_COLORS[nb.node?.type] || '#94a3b8'}`, transition: 'background 0.15s' }}
                      onMouseEnter={e => e.currentTarget.style.background = '#f1f5f9'}
                      onMouseLeave={e => e.currentTarget.style.background = '#f8fafc'}>
                      <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)', fontWeight: 500 }}>
                        {data.node.label} →
                      </div>
                      <div style={{ fontSize: '0.72rem', color: 'var(--primary)', fontWeight: 700, marginBottom: '0.15rem' }}>
                        {RELATION_TR[nb.edge?.relation] || nb.edge?.relation}
                        {nb.direction === 'outgoing' ? ' →' : ' ←'}
                      </div>
                      <div style={{ fontSize: '0.82rem', fontWeight: 600 }}>{nb.node?.label}</div>
                      <Badge type={nb.node?.type} />
                    </div>
                  ))
                ).slice(0, 30)}
              </div>
            </>
          )}
        </div>

        {/* Path Finder */}
        <div style={{ padding: '1rem', borderTop: '1px solid var(--border)', background: 'var(--bg)', flexShrink: 0 }}>
          <button onClick={() => setPathMode(!pathMode)}
            style={{ width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.4rem', padding: '0.6rem', borderRadius: '8px', border: `1px solid ${pathMode ? 'var(--primary)' : 'var(--border)'}`, background: pathMode ? 'var(--primary)' : 'white', color: pathMode ? 'white' : 'var(--text-main)', cursor: 'pointer', fontSize: '0.85rem', fontWeight: 600 }}>
            <GitBranch size={14} /> Yol Bulucu
          </button>
          {pathMode && (
            <div style={{ marginTop: '0.75rem', display: 'flex', flexDirection: 'column', gap: '0.6rem' }}>
              <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', background: 'rgba(0,0,0,0.03)', padding: '0.5rem', borderRadius: '6px' }}>Yol bulmak için iki düğüm aratın veya grafikte seçin.</div>
              
              {!pathFrom ? (
                <div style={{ position: 'relative' }}>
                  <Search size={12} style={{ position: 'absolute', left: 8, top: '50%', transform: 'translateY(-50%)', color: 'var(--text-muted)' }} />
                  <input 
                    placeholder="Başlangıç düğümü ara..."
                    onChange={(e) => handleSearch(e.target.value)}
                    style={{ width: '100%', padding: '0.4rem 0.5rem 0.4rem 1.75rem', fontSize: '0.75rem', borderRadius: '6px', border: '1px solid var(--border)' }}
                  />
                  {search && searchResults.length > 0 && (
                    <div style={{ position: 'absolute', bottom: '100%', left: 0, right: 0, background: 'white', border: '1px solid var(--border)', borderRadius: '6px', zIndex: 20, maxHeight: '150px', overflowY: 'auto', boxShadow: '0 -4px 6px -1px rgba(0,0,0,0.1)' }}>
                      {searchResults.map(n => (
                        <div key={n.node_id} onClick={() => { setPathFrom(n); setSearch(''); setSearchResults([]); }} style={{ padding: '0.4rem 0.6rem', cursor: 'pointer', borderBottom: '1px solid var(--border)', fontSize: '0.75rem' }}>{n.label}</div>
                      ))}
                    </div>
                  )}
                </div>
              ) : (
                <div style={{ fontSize: '0.8rem', padding: '0.4rem 0.6rem', background: '#e0f2fe', borderRadius: '6px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>Bkz: <b>{pathFrom.label}</b></span>
                  <X size={14} style={{ cursor: 'pointer', flexShrink: 0 }} onClick={() => setPathFrom(null)} />
                </div>
              )}

              {!pathTo ? (
                <div style={{ position: 'relative' }}>
                  <Search size={12} style={{ position: 'absolute', left: 8, top: '50%', transform: 'translateY(-50%)', color: 'var(--text-muted)' }} />
                  <input 
                    placeholder="Hedef düğüm ara..."
                    onChange={(e) => handleSearch(e.target.value)}
                    style={{ width: '100%', padding: '0.4rem 0.5rem 0.4rem 1.75rem', fontSize: '0.75rem', borderRadius: '6px', border: '1px solid var(--border)' }}
                  />
                  {search && searchResults.length > 0 && (
                    <div style={{ position: 'absolute', bottom: '100%', left: 0, right: 0, background: 'white', border: '1px solid var(--border)', borderRadius: '6px', zIndex: 20, maxHeight: '150px', overflowY: 'auto', boxShadow: '0 -4px 6px -1px rgba(0,0,0,0.1)' }}>
                      {searchResults.map(n => (
                        <div key={n.node_id} onClick={() => { setPathTo(n); setSearch(''); setSearchResults([]); }} style={{ padding: '0.4rem 0.6rem', cursor: 'pointer', borderBottom: '1px solid var(--border)', fontSize: '0.75rem' }}>{n.label}</div>
                      ))}
                    </div>
                  )}
                </div>
              ) : (
                <div style={{ fontSize: '0.8rem', padding: '0.4rem 0.6rem', background: '#dcfce7', borderRadius: '6px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>Hedef: <b>{pathTo.label}</b></span>
                  <X size={14} style={{ cursor: 'pointer', flexShrink: 0 }} onClick={() => setPathTo(null)} />
                </div>
              )}

              {pathFrom && pathTo && (
                <button onClick={findPath} disabled={pathLoading}
                  style={{ padding: '0.6rem', background: 'var(--primary)', color: 'white', border: 'none', borderRadius: '8px', cursor: 'pointer', fontWeight: 700 }}>
                  {pathLoading ? 'Aranıyor...' : 'Yol Bul'}
                </button>
              )}
              {pathResult && (
                <div style={{ fontSize: '0.78rem', background: 'white', padding: '0.75rem', borderRadius: '8px', border: '1px solid var(--border)', borderLeft: '4px solid var(--primary)' }}>
                  {pathResult.found
                    ? <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.2rem' }}>{pathResult.path.map((s, i) => <span key={i}>{i > 0 && <span style={{ color: '#94a3b8' }}> → </span>}<span style={{ background: '#f1f5f9', padding: '0.1rem 0.3rem', borderRadius: '4px' }}>{s}</span></span>)}</div>
                    : <span style={{ color: '#ef4444' }}>Yol bulunamadı.</span>
                  }
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* ── Graph Canvas ─────────────────────────────────── */}
      <div ref={contRef} style={{ flex: 1, position: 'relative', overflow: 'hidden' }}>
        {loading && (
          <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', flexDirection: 'column', gap: '0.75rem', background: 'var(--bg)', zIndex: 10 }}>
            <Loader size={32} style={{ opacity: 0.4, animation: 'spin 1s linear infinite' }} />
            <div style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>Bilgi grafiği yükleniyor...</div>
          </div>
        )}
        {error && (
          <div style={{ position: 'absolute', top: 16, left: 16, background: '#fef2f2', color: '#ef4444', padding: '0.75rem 1rem', borderRadius: '8px', border: '1px solid #fee2e2' }}>
            {error}
          </div>
        )}
        {!loading && graphData.nodes.length === 0 && (
          <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', flexDirection: 'column', gap: '1rem' }}>
            <Share2 size={48} style={{ opacity: 0.15 }} />
            <div style={{ color: 'var(--text-muted)', fontSize: '1rem' }}>Bilgi grafiği henüz oluşturulmamış.</div>
            <button onClick={handleRebuild} disabled={rebuilding} style={{ padding: '0.75rem 1.5rem', background: 'var(--primary)', color: 'white', border: 'none', borderRadius: '10px', fontWeight: 700, cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <RefreshCw size={16} /> KG Oluştur
            </button>
          </div>
        )}

        {!loading && graphData.nodes.length > 0 && (
          <ForceGraph2D
            ref={fgRef}
            width={dimensions.width}
            height={dimensions.height}
            graphData={graphData}
            nodeCanvasObject={(node, ctx, globalScale) => {
              const isSelected = selectedNodes.some(sn => (sn.node_id === node.id || sn.id === node.id));
              const r = isSelected ? 10 : 6;
              ctx.beginPath();
              ctx.arc(node.x, node.y, r, 0, 2 * Math.PI);
              ctx.fillStyle = NODE_COLORS[node.type] || '#94a3b8';
              ctx.fill();
              if (isSelected) {
                ctx.strokeStyle = '#fff';
                ctx.lineWidth = 3;
                ctx.stroke();
                // Outer glow
                ctx.strokeStyle = 'rgba(59, 130, 246, 0.5)';
                ctx.lineWidth = 2;
                ctx.arc(node.x, node.y, r + 4, 0, 2 * Math.PI);
                ctx.stroke();
              }
              if (globalScale > 1.2) {
                ctx.font = `${Math.min(12 / globalScale, 10)}px Inter, sans-serif`;
                ctx.fillStyle = '#1e293b';
                ctx.textAlign = 'center';
                ctx.fillText(node.label?.slice(0, 20) || '', node.x, node.y + r + 10);
              }
            }}
            nodeCanvasObjectMode={() => 'replace'}
            linkColor={() => 'rgba(100,116,139,0.25)'}
            linkWidth={1}
            linkDirectionalArrowLength={4}
            linkDirectionalArrowRelPos={1}
            linkLabel={link => RELATION_TR[link.relation] || link.relation || ''}
            onNodeClick={(node, event) => {
              if (pathMode) {
                if (!pathFrom) setPathFrom(node);
                else if (!pathTo) setPathTo(node);
              } else {
                selectNodes([node], event.shiftKey);
              }
            }}
            onNodeRightClick={(node) => {
              if (pathMode) {
                setPathFrom(node);
                setPathTo(null);
                setPathResult(null);
              }
            }}
            cooldownTicks={120}
            d3AlphaDecay={0.02}
            d3VelocityDecay={0.3}
          />
        )}
        
        {/* Controls Overlay */}
        <div style={{ position: 'absolute', top: 16, right: 16, display: 'flex', flexDirection: 'column', gap: '0.5rem', zIndex: 10 }}>
          <button onClick={() => fgRef.current?.zoomToFit(800, 50)} title="Sığdır"
            style={{ width: '36px', height: '36px', borderRadius: '8px', background: 'white', border: '1px solid var(--border)', color: 'var(--text-main)', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', boxShadow: '0 2px 6px rgba(0,0,0,0.1)' }}>
            <Share2 size={16} />
          </button>
        </div>

        {/* Legend */}
        {!loading && graphData.nodes.length > 0 && (
          <div style={{ position: 'absolute', bottom: 16, right: 16, background: 'var(--card-bg)', padding: '0.75rem 1rem', borderRadius: '10px', border: '1px solid var(--border)', fontSize: '0.75rem', boxShadow: '0 2px 8px rgba(0,0,0,0.08)' }}>
            <div style={{ fontWeight: 700, marginBottom: '0.4rem', fontSize: '0.7rem', textTransform: 'uppercase', color: 'var(--text-muted)' }}>Lejant</div>
            {Object.entries(NODE_TYPE_TR).map(([key, label]) => (
              <div key={key} style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', marginBottom: '0.2rem' }}>
                <div style={{ width: 10, height: 10, borderRadius: '50%', background: NODE_COLORS[key], flexShrink: 0 }} />
                <span style={{ color: 'var(--text-main)' }}>{label}</span>
              </div>
            ))}
          </div>
        )}
      </div>

      <style>{`
        @keyframes spin { to { transform: rotate(360deg); } }
        .spin { animation: spin 1s linear infinite; }
      `}</style>
    </div>
  );
}

export default KnowledgeGraph;
