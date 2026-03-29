import React, { useState, useEffect, useCallback } from 'react';
import ForceGraph2D from 'react-force-graph-2d';

function KnowledgeGraph() {
  const [data, setData] = useState({ nodes: [], links: [] });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [dimensions, setDimensions] = useState({ width: 800, height: 600 });

  useEffect(() => {
    // Update dimensions on resize
    const container = document.getElementById('kg-container');
    if (container) {
      setDimensions({ width: container.clientWidth, height: container.clientHeight });
    }
    const handleResize = () => {
      if (container) {
        setDimensions({ width: container.clientWidth, height: container.clientHeight });
      }
    };
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  useEffect(() => {
    async function fetchKG() {
      try {
        const response = await fetch('/api/kg', {
          headers: {
            'Authorization': `Bearer ${localStorage.getItem('token')}`
          }
        });
        if (!response.ok) throw new Error('Veri çekilemedi.');
        const kgData = await response.json();
        
        // Transform edges format to match react-force-graph
        const links = kgData.edges.map(e => ({
          source: e.source,
          target: e.target,
          name: e.relation
        }));
        
        setData({ nodes: kgData.nodes, links });
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    }
    fetchKG();
  }, []);

  const getNodeColor = (node) => {
    switch(node.type) {
      case 'RULE': return '#3b82f6'; // Blue
      case 'DRUG': return '#10b981'; // Green
      case 'CONDITION':
      case 'DIAGNOSIS': return '#ef4444'; // Red
      default: return '#94a3b8'; // Slate
    }
  };

  return (
    <div id="kg-container" style={{ width: '100%', height: '100%', position: 'relative' }}>
      {loading && <div style={{ position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)' }}>Yükleniyor...</div>}
      {error && <div style={{ color: 'red', position: 'absolute', top: 20, left: 20 }}>{error}</div>}
      
      {!loading && !error && (
        <ForceGraph2D
          width={dimensions.width}
          height={dimensions.height}
          graphData={data}
          nodeLabel="label"
          nodeColor={getNodeColor}
          nodeRelSize={6}
          linkColor={() => '#cbd5e1'}
          linkDirectionalArrowLength={3}
          linkDirectionalArrowRelPos={1}
          onNodeClick={(node) => {
            alert(`Tip: ${node.type}\nBaşlık: ${node.label}\n\nİçerik:\n${node.text || 'Detay yok.'}`);
          }}
        />
      )}
      
      <div style={{ position: 'absolute', bottom: 20, right: 20, background: 'rgba(255,255,255,0.9)', padding: '10px', borderRadius: '8px', border: '1px solid #e2e8f0', fontSize: '0.8rem' }}>
        <div style={{ fontWeight: 'bold', marginBottom: '5px' }}>Lejant</div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '5px' }}><div style={{ width: 10, height: 10, background: '#3b82f6', borderRadius: '50%' }}></div> SUT Kuralı</div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '5px' }}><div style={{ width: 10, height: 10, background: '#10b981', borderRadius: '50%' }}></div> İlaç / Etken Madde</div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '5px' }}><div style={{ width: 10, height: 10, background: '#ef4444', borderRadius: '50%' }}></div> Hastalık / Durum</div>
      </div>
    </div>
  );
}

export default KnowledgeGraph;
