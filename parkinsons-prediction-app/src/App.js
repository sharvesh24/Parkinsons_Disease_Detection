import React, { useState, useRef, useEffect } from 'react';
import { 
  Activity, User, FileText, Settings, Moon, Sun, Download, 
  ChevronLeft, ChevronRight, Check, AlertTriangle, Info, 
  RotateCcw, Menu, X, Eye, BarChart3, Calendar, Clock
} from 'lucide-react';

const ParkinsonsApp = () => {
  const [screen, setScreen] = useState('home');
  const [darkMode, setDarkMode] = useState(true);
  const [menuOpen, setMenuOpen] = useState(false);
  const [drawingPaths, setDrawingPaths] = useState({
    circle: [],
    spiral: [],
    meander: []
  });
  const [currentPattern, setCurrentPattern] = useState('circle');
  const [patientData, setPatientData] = useState({
    age: '',
    gender: '',
    weight: '',
    height: '',
    writingHand: 'right',
    smoker: false
  });
  const [results, setResults] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [assessmentHistory, setAssessmentHistory] = useState([]);
  const [selectedReport, setSelectedReport] = useState(null);

  const canvasRef = useRef(null);
  const [isDrawing, setIsDrawing] = useState(false);

  const theme = {
    bg: darkMode ? '#0f172a' : '#f8fafc',
    card: darkMode ? '#1e293b' : '#ffffff',
    text: darkMode ? '#f1f5f9' : '#0f172a',
    textSecondary: darkMode ? '#94a3b8' : '#64748b',
    border: darkMode ? '#334155' : '#e2e8f0',
    primary: '#3b82f6',
    success: '#10b981',
    warning: '#f59e0b',
    danger: '#ef4444',
    accent: '#8b5cf6'
  };

  const patterns = ['circle', 'spiral', 'meander'];
  const patternInfo = {
    circle: {
      name: 'Circle Drawing',
      description: 'Draw a complete circle starting and ending at the same point',
      difficulty: 'Medium',
      color: theme.primary,
      icon: '⭕'
    },
    spiral: {
      name: 'Spiral Drawing',
      description: 'Start from center and draw outward in a spiral pattern',
      difficulty: 'Hard',
      color: theme.warning,
      icon: '🌀'
    },
    meander: {
      name: 'Wave Pattern',
      description: 'Draw a continuous wavy line from left to right',
      difficulty: 'Easy',
      color: theme.success,
      icon: '〰️'
    }
  };

  const handleMouseDown = (e) => {
    setIsDrawing(true);
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    setDrawingPaths(prev => ({
      ...prev,
      [currentPattern]: [...prev[currentPattern], { x, y }]
    }));
  };

  const handleTouchStart = (e) => {
    e.preventDefault();
    setIsDrawing(true);
    const rect = e.currentTarget.getBoundingClientRect();
    const touch = e.touches[0];
    const x = touch.clientX - rect.left;
    const y = touch.clientY - rect.top;
    setDrawingPaths(prev => ({
      ...prev,
      [currentPattern]: [...prev[currentPattern], { x, y }]
    }));
  };

  const handleMouseMove = (e) => {
    if (!isDrawing) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    setDrawingPaths(prev => ({
      ...prev,
      [currentPattern]: [...prev[currentPattern], { x, y }]
    }));
  };

  const handleTouchMove = (e) => {
    e.preventDefault();
    if (!isDrawing) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const touch = e.touches[0];
    const x = touch.clientX - rect.left;
    const y = touch.clientY - rect.top;
    setDrawingPaths(prev => ({
      ...prev,
      [currentPattern]: [...prev[currentPattern], { x, y }]
    }));
  };

  const handleMouseUp = () => {
    setIsDrawing(false);
  };

  const clearCanvas = () => {
    setDrawingPaths(prev => ({
      ...prev,
      [currentPattern]: []
    }));
  };

  const getCompletedPatterns = () => {
    return Object.values(drawingPaths).filter(path => path.length > 0).length;
  };

  const analyzeData = async () => {
  console.log('🟢 analyzeData function called!');
  console.log('Can analyze?', canAnalyze());
  console.log('Drawing paths:', drawingPaths);
  console.log('Patient data:', patientData);
  
  setIsAnalyzing(true);
  
  try {
    const API_URL = 'https://parkinsons-detection-api-ljeo.onrender.com';
    console.log('🔵 Starting API calls to:', API_URL);
    
    // 1. Analyze drawings
    console.log('📤 Sending drawing data...');
    const drawingResponse = await fetch(`${API_URL}/api/analyze-drawings`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(drawingPaths)
    });
    console.log('📥 Drawing response status:', drawingResponse.status);
    const drawingResult = await drawingResponse.json();
    console.log('✅ Drawing result:', drawingResult);
    
    // 2. Analyze clinical data
    console.log('📤 Sending clinical data...');
    const clinicalResponse = await fetch(`${API_URL}/api/analyze-clinical`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(patientData)
    });
    console.log('📥 Clinical response status:', clinicalResponse.status);
    const clinicalResult = await clinicalResponse.json();
    console.log('✅ Clinical result:', clinicalResult);
    
    // 3. Get combined analysis
    console.log('📤 Sending combined data...');
    const combinedResponse = await fetch(`${API_URL}/api/analyze-combined`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        drawingPrediction: drawingResult,
        clinicalPrediction: clinicalResult
      })
    });
    console.log('📥 Combined response status:', combinedResponse.status);
    const combinedResult = await combinedResponse.json();
    console.log('✅ Combined result:', combinedResult);
    
    // Use real results
    const newAssessment = {
      id: combinedResult.assessment_id,
      date: new Date().toISOString(),
      result: combinedResult.result,
      confidence: combinedResult.confidence,
      drawingScore: drawingResult.confidence,
      clinicalScore: clinicalResult.confidence,
      patientData: { ...patientData }
    };
    
    console.log('📊 Final assessment:', newAssessment);
    
    setResults(newAssessment);
    setAssessmentHistory(prev => [newAssessment, ...prev]);
    setScreen('results');
    console.log('🎉 Screen changed to results');
    
  } catch (error) {
    console.error('❌ API Error:', error);
    console.error('Error stack:', error.stack);
    alert('Failed to analyze. Error: ' + error.message);
  } finally {
    console.log('🔄 Setting isAnalyzing to false');
    setIsAnalyzing(false);
  }
};

  const generatePDF = async (assessment) => {
    alert(`PDF Report generated for Assessment ID: ${assessment.id}\n\nIn production, this would download a PDF file with complete medical report.`);
  };

  const drawPath = (ctx, path, color) => {
    if (path.length < 2) return;
    ctx.strokeStyle = color;
    ctx.lineWidth = 3;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.beginPath();
    ctx.moveTo(path[0].x, path[0].y);
    for (let i = 1; i < path.length; i++) {
      ctx.lineTo(path[i].x, path[i].y);
    }
    ctx.stroke();
  };

  useEffect(() => {
    if (canvasRef.current) {
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      
      ctx.strokeStyle = patternInfo[currentPattern].color;
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 5]);
      ctx.globalAlpha = 0.3;
      
      if (currentPattern === 'circle') {
        ctx.beginPath();
        ctx.arc(200, 200, 80, 0, Math.PI * 2);
        ctx.stroke();
      } else if (currentPattern === 'spiral') {
        ctx.beginPath();
        let angle = 0;
        let radius = 5;
        ctx.moveTo(200, 200);
        for (let i = 0; i < 150; i++) {
          angle += 0.25;
          radius += 0.6;
          const x = 200 + radius * Math.cos(angle);
          const y = 200 + radius * Math.sin(angle);
          ctx.lineTo(x, y);
        }
        ctx.stroke();
      } else if (currentPattern === 'meander') {
        ctx.beginPath();
        for (let x = 20; x < 380; x += 3) {
          const y = 200 + 30 * Math.sin(x * 0.03);
          if (x === 20) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
        ctx.stroke();
      }
      
      ctx.globalAlpha = 1;
      ctx.setLineDash([]);
      
      if (drawingPaths[currentPattern].length > 0) {
        drawPath(ctx, drawingPaths[currentPattern], patternInfo[currentPattern].color);
      }
    }
  }, [drawingPaths, currentPattern, darkMode]);

  const Header = ({ title, onBack, rightAction }) => (
    <div style={{
      padding: '16px',
      backgroundColor: theme.card,
      borderBottom: `1px solid ${theme.border}`,
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between'
    }}>
      <button 
        onClick={onBack}
        style={{
          background: 'none',
          border: 'none',
          color: theme.text,
          cursor: 'pointer',
          padding: '8px',
          borderRadius: '8px'
        }}
      >
        <ChevronLeft size={24} />
      </button>
      <h2 style={{ margin: 0, fontSize: '16px', fontWeight: '600', color: theme.text }}>{title}</h2>
      <div style={{ width: '40px' }}>{rightAction}</div>
    </div>
  );

  if (screen === 'home') {
    return (
      <div style={{
        minHeight: '100vh',
        backgroundColor: theme.bg,
        color: theme.text,
        fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif'
      }}>
        <div style={{ maxWidth: '480px', margin: '0 auto', position: 'relative' }}>
          <div style={{
            padding: '16px',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            borderBottom: `1px solid ${theme.border}`
          }}>
            <button
              onClick={() => setMenuOpen(!menuOpen)}
              style={{
                background: 'none',
                border: 'none',
                color: theme.text,
                cursor: 'pointer',
                padding: '8px',
                borderRadius: '8px'
              }}
            >
              <Menu size={24} />
            </button>
            <h1 style={{ margin: 0, fontSize: '18px', fontWeight: '700' }}>NeuroCheck</h1>
            <button
              onClick={() => setDarkMode(!darkMode)}
              style={{
                background: 'none',
                border: 'none',
                color: theme.text,
                cursor: 'pointer',
                padding: '8px',
                borderRadius: '8px'
              }}
            >
              {darkMode ? <Sun size={20} /> : <Moon size={20} />}
            </button>
          </div>

          <div style={{ padding: '32px 24px', textAlign: 'center' }}>
            <div style={{
              width: '80px',
              height: '80px',
              backgroundColor: `${theme.primary}20`,
              borderRadius: '20px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              margin: '0 auto 20px',
              border: `2px solid ${theme.primary}`
            }}>
              <Activity size={40} color={theme.primary} />
            </div>
            <h2 style={{ fontSize: '24px', fontWeight: '700', marginBottom: '8px' }}>
              Parkinson's Screening
            </h2>
            <p style={{ color: theme.textSecondary, fontSize: '14px', margin: 0 }}>
              AI-powered early detection assessment
            </p>
          </div>

          <div style={{ padding: '0 24px 24px' }}>
            <button
              onClick={() => setScreen('drawing')}
              style={{
                width: '100%',
                backgroundColor: theme.card,
                padding: '20px',
                borderRadius: '16px',
                border: `2px solid ${theme.border}`,
                marginBottom: '16px',
                cursor: 'pointer',
                textAlign: 'left',
                display: 'flex',
                alignItems: 'flex-start',
                transition: 'all 0.2s'
              }}
            >
              <div style={{
                width: '48px',
                height: '48px',
                backgroundColor: `${theme.primary}15`,
                borderRadius: '12px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                marginRight: '16px',
                flexShrink: 0
              }}>
                <Activity size={24} color={theme.primary} />
              </div>
              <div style={{ flex: 1 }}>
                <h3 style={{ fontSize: '16px', fontWeight: '600', margin: '0 0 4px 0', color: theme.text }}>
                  Motor Assessment
                </h3>
                <p style={{ fontSize: '13px', color: theme.textSecondary, margin: '0 0 8px 0' }}>
                  Hand movement and coordination analysis
                </p>
                {getCompletedPatterns() > 0 && (
                  <div style={{
                    display: 'inline-block',
                    padding: '4px 12px',
                    backgroundColor: `${theme.primary}20`,
                    borderRadius: '12px',
                    fontSize: '12px',
                    color: theme.primary,
                    fontWeight: '500'
                  }}>
                    {getCompletedPatterns()}/3 patterns completed
                  </div>
                )}
              </div>
              {getCompletedPatterns() === 3 && (
                <Check size={20} color={theme.success} />
              )}
            </button>

            <button
              onClick={() => setScreen('clinical')}
              style={{
                width: '100%',
                backgroundColor: theme.card,
                padding: '20px',
                borderRadius: '16px',
                border: `2px solid ${theme.border}`,
                marginBottom: '16px',
                cursor: 'pointer',
                textAlign: 'left',
                display: 'flex',
                alignItems: 'flex-start'
              }}
            >
              <div style={{
                width: '48px',
                height: '48px',
                backgroundColor: `${theme.warning}15`,
                borderRadius: '12px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                marginRight: '16px',
                flexShrink: 0
              }}>
                <User size={24} color={theme.warning} />
              </div>
              <div style={{ flex: 1 }}>
                <h3 style={{ fontSize: '16px', fontWeight: '600', margin: '0 0 4px 0', color: theme.text }}>
                  Clinical Assessment
                </h3>
                <p style={{ fontSize: '13px', color: theme.textSecondary, margin: '0 0 8px 0' }}>
                  Demographic and medical information
                </p>
                {patientData.age && patientData.gender && (
                  <div style={{
                    display: 'inline-block',
                    padding: '4px 12px',
                    backgroundColor: `${theme.warning}20`,
                    borderRadius: '12px',
                    fontSize: '12px',
                    color: theme.warning,
                    fontWeight: '500'
                  }}>
                    Information provided
                  </div>
                )}
              </div>
              {patientData.age && patientData.gender && (
                <Check size={20} color={theme.success} />
              )}
            </button>

            <button
              onClick={() => setScreen('history')}
              style={{
                width: '100%',
                backgroundColor: theme.card,
                padding: '20px',
                borderRadius: '16px',
                border: `2px solid ${theme.border}`,
                marginBottom: '24px',
                cursor: 'pointer',
                textAlign: 'left',
                display: 'flex',
                alignItems: 'flex-start'
              }}
            >
              <div style={{
                width: '48px',
                height: '48px',
                backgroundColor: `${theme.accent}15`,
                borderRadius: '12px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                marginRight: '16px',
                flexShrink: 0
              }}>
                <BarChart3 size={24} color={theme.accent} />
              </div>
              <div style={{ flex: 1 }}>
                <h3 style={{ fontSize: '16px', fontWeight: '600', margin: '0 0 4px 0', color: theme.text }}>
                  Assessment History
                </h3>
                <p style={{ fontSize: '13px', color: theme.textSecondary, margin: 0 }}>
                  View past assessments and reports
                </p>
              </div>
              <ChevronRight size={20} color={theme.textSecondary} />
            </button>

            <button
              onClick={analyzeData}
              disabled={!canAnalyze()}
              style={{
                width: '100%',
                padding: '16px',
                borderRadius: '12px',
                border: 'none',
                backgroundColor: canAnalyze() ? theme.primary : theme.border,
                color: canAnalyze() ? '#ffffff' : theme.textSecondary,
                fontSize: '16px',
                fontWeight: '600',
                cursor: canAnalyze() ? 'pointer' : 'not-allowed',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                transition: 'all 0.2s'
              }}
            >
              <Activity size={20} style={{ marginRight: '8px' }} />
              Analyze Assessment
            </button>
          </div>

          <div style={{
            margin: '0 24px 24px',
            padding: '16px',
            backgroundColor: `${theme.danger}15`,
            border: `1px solid ${theme.danger}40`,
            borderRadius: '12px'
          }}>
            <div style={{ display: 'flex', alignItems: 'flex-start' }}>
              <Info size={18} color={theme.danger} style={{ marginRight: '12px', marginTop: '2px', flexShrink: 0 }} />
              <div style={{ fontSize: '12px', lineHeight: '1.5' }}>
                <p style={{ fontWeight: '600', color: theme.danger, margin: '0 0 4px 0' }}>
                  Medical Disclaimer
                </p>
                <p style={{ color: theme.textSecondary, margin: 0 }}>
                  This tool is for screening purposes only. Results are not medical diagnoses. 
                  Consult healthcare professionals for medical advice.
                </p>
              </div>
            </div>
          </div>

          {menuOpen && (
            <>
              <div
                onClick={() => setMenuOpen(false)}
                style={{
                  position: 'fixed',
                  top: 0,
                  left: 0,
                  right: 0,
                  bottom: 0,
                  backgroundColor: 'rgba(0, 0, 0, 0.5)',
                  zIndex: 999
                }}
              />
              <div style={{
                position: 'fixed',
                top: 0,
                left: 0,
                bottom: 0,
                width: '280px',
                backgroundColor: theme.card,
                zIndex: 1000,
                padding: '24px',
                boxShadow: '2px 0 10px rgba(0, 0, 0, 0.1)'
              }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '32px' }}>
                  <h3 style={{ margin: 0, fontSize: '18px', fontWeight: '700' }}>Menu</h3>
                  <button
                    onClick={() => setMenuOpen(false)}
                    style={{ background: 'none', border: 'none', color: theme.text, cursor: 'pointer' }}
                  >
                    <X size={24} />
                  </button>
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                  <button
                    onClick={() => { setScreen('home'); setMenuOpen(false); }}
                    style={{
                      padding: '12px 16px',
                      backgroundColor: theme.bg,
                      border: 'none',
                      borderRadius: '8px',
                      color: theme.text,
                      textAlign: 'left',
                      cursor: 'pointer',
                      fontSize: '14px',
                      fontWeight: '500'
                    }}
                  >
                    Home
                  </button>
                  <button
                    onClick={() => { setScreen('history'); setMenuOpen(false); }}
                    style={{
                      padding: '12px 16px',
                      backgroundColor: theme.bg,
                      border: 'none',
                      borderRadius: '8px',
                      color: theme.text,
                      textAlign: 'left',
                      cursor: 'pointer',
                      fontSize: '14px',
                      fontWeight: '500'
                    }}
                  >
                    History
                  </button>
                  <button
                    onClick={() => { setScreen('about'); setMenuOpen(false); }}
                    style={{
                      padding: '12px 16px',
                      backgroundColor: theme.bg,
                      border: 'none',
                      borderRadius: '8px',
                      color: theme.text,
                      textAlign: 'left',
                      cursor: 'pointer',
                      fontSize: '14px',
                      fontWeight: '500'
                    }}
                  >
                    About
                  </button>
                </div>
              </div>
            </>
          )}
        </div>
      </div>
    );
  }

  if (screen === 'drawing') {
    const currentIndex = patterns.indexOf(currentPattern);
    
    return (
      <div style={{ minHeight: '100vh', backgroundColor: theme.bg, color: theme.text }}>
        <div style={{ maxWidth: '480px', margin: '0 auto' }}>
          <Header title="Motor Assessment" onBack={() => setScreen('home')} />

          <div style={{ padding: '24px' }}>
            <div style={{
              backgroundColor: theme.card,
              borderRadius: '16px',
              padding: '20px',
              marginBottom: '20px',
              border: `1px solid ${theme.border}`
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '12px' }}>
                <div>
                  <div style={{ fontSize: '24px', marginBottom: '4px' }}>{patternInfo[currentPattern].icon}</div>
                  <h3 style={{ fontSize: '18px', fontWeight: '600', margin: '0 0 4px 0' }}>
                    {patternInfo[currentPattern].name}
                  </h3>
                  <p style={{ fontSize: '13px', color: theme.textSecondary, margin: 0 }}>
                    {patternInfo[currentPattern].description}
                  </p>
                </div>
                <span style={{
                  padding: '6px 12px',
                  borderRadius: '20px',
                  fontSize: '12px',
                  fontWeight: '500',
                  backgroundColor: `${patternInfo[currentPattern].color}20`,
                  color: patternInfo[currentPattern].color
                }}>
                  {patternInfo[currentPattern].difficulty}
                </span>
              </div>
              <div style={{ fontSize: '12px', color: theme.textSecondary }}>
                Pattern {currentIndex + 1} of {patterns.length}
              </div>
            </div>

            <div style={{
              backgroundColor: theme.card,
              borderRadius: '16px',
              overflow: 'hidden',
              marginBottom: '16px',
              border: `1px solid ${theme.border}`
            }}>
              <canvas
                ref={canvasRef}
                width={400}
                height={400}
                onMouseDown={handleMouseDown}
                onMouseMove={handleMouseMove}
                onMouseUp={handleMouseUp}
                onMouseLeave={handleMouseUp}
                onTouchStart={handleTouchStart}
                onTouchMove={handleTouchMove}
                onTouchEnd={handleMouseUp}
                style={{ width: '100%', cursor: 'crosshair', display: 'block', touchAction: 'none' }}
              />
              <div style={{ padding: '16px' }}>
                <button
                  onClick={clearCanvas}
                  style={{
                    width: '100%',
                    padding: '12px',
                    backgroundColor: theme.danger,
                    color: '#ffffff',
                    border: 'none',
                    borderRadius: '8px',
                    fontSize: '14px',
                    fontWeight: '600',
                    cursor: 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center'
                  }}
                >
                  <RotateCcw size={16} style={{ marginRight: '8px' }} />
                  Clear Drawing
                </button>
              </div>
            </div>

            <div style={{ display: 'flex', gap: '8px', marginBottom: '20px' }}>
              {patterns.map((pattern) => (
                <button
                  key={pattern}
                  onClick={() => setCurrentPattern(pattern)}
                  style={{
                    flex: 1,
                    height: '8px',
                    border: 'none',
                    borderRadius: '4px',
                    backgroundColor: patternInfo[pattern].color,
                    opacity: pattern === currentPattern ? 1 : (drawingPaths[pattern].length > 0 ? 0.5 : 0.2),
                    cursor: 'pointer',
                    transition: 'opacity 0.2s'
                  }}
                />
              ))}
            </div>

            <div style={{ display: 'flex', gap: '12px' }}>
              <button
                onClick={() => {
                  const prevIndex = Math.max(0, currentIndex - 1);
                  setCurrentPattern(patterns[prevIndex]);
                }}
                disabled={currentIndex === 0}
                style={{
                  flex: 1,
                  padding: '14px',
                  borderRadius: '12px',
                  border: 'none',
                  backgroundColor: currentIndex === 0 ? theme.border : theme.card,
                  color: currentIndex === 0 ? theme.textSecondary : theme.text,
                  fontSize: '14px',
                  fontWeight: '600',
                  cursor: currentIndex === 0 ? 'not-allowed' : 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center'
                }}
              >
                <ChevronLeft size={18} style={{ marginRight: '4px' }} />
                Previous
              </button>

              <button
                onClick={() => {
                  if (currentIndex < patterns.length - 1) {
                    setCurrentPattern(patterns[currentIndex + 1]);
                  } else {
                    setScreen('home');
                  }
                }}
                style={{
                  flex: 1,
                  padding: '14px',
                  backgroundColor: theme.primary,
                  color: '#ffffff',
                  border: 'none',
                  borderRadius: '12px',
                  fontSize: '14px',
                  fontWeight: '600',
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center'
                }}
              >
                {currentIndex < patterns.length - 1 ? (
                  <>
                    Next
                    <ChevronRight size={18} style={{ marginLeft: '4px' }} />
                  </>
                ) : (
                  <>
                    <Check size={18} style={{ marginRight: '8px' }} />
                    Complete
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (screen === 'clinical') {
    const bmi = patientData.weight && patientData.height 
      ? (parseFloat(patientData.weight) / Math.pow(parseFloat(patientData.height) / 100, 2)).toFixed(1)
      : null;

    return (
      <div style={{ minHeight: '100vh', backgroundColor: theme.bg, color: theme.text }}>
        <div style={{ maxWidth: '480px', margin: '0 auto' }}>
          <Header title="Clinical Assessment" onBack={() => setScreen('home')} />

          <div style={{ padding: '24px' }}>
            <div style={{
              backgroundColor: theme.card,
              borderRadius: '16px',
              padding: '24px',
              border: `1px solid ${theme.border}`
            }}>
              <h3 style={{ fontSize: '16px', fontWeight: '600', marginBottom: '20px' }}>
                Patient Information
              </h3>
              
              <div style={{ marginBottom: '20px' }}>
                <label style={{ display: 'block', fontSize: '13px', fontWeight: '500', marginBottom: '8px' }}>
                  Age <span style={{ color: theme.danger }}>*</span>
                </label>
                <input
                  type="number"
                  value={patientData.age}
                  onChange={(e) => setPatientData(prev => ({ ...prev, age: e.target.value }))}
                  placeholder="Enter your age"
                  style={{
                    width: '100%',
                    padding: '12px 16px',
                    backgroundColor: theme.bg,
                    border: `1px solid ${theme.border}`,
                    borderRadius: '8px',
                    color: theme.text,
                    fontSize: '14px',
                    boxSizing: 'border-box'
                  }}
                />
              </div>

              <div style={{ marginBottom: '20px' }}>
                <label style={{ display: 'block', fontSize: '13px', fontWeight: '500', marginBottom: '8px' }}>
                  Gender <span style={{ color: theme.danger }}>*</span>
                </label>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '8px' }}>
                  {['Male', 'Female', 'Other'].map((gender) => (
                    <button
                      key={gender}
                      onClick={() => setPatientData(prev => ({ ...prev, gender: gender.toLowerCase() }))}
                      style={{
                        padding: '12px',
                        borderRadius: '8px',
                        border: 'none',
                        backgroundColor: patientData.gender === gender.toLowerCase() ? theme.primary : theme.bg,
                        color: patientData.gender === gender.toLowerCase() ? '#ffffff' : theme.text,
                        fontSize: '14px',
                        fontWeight: '500',
                        cursor: 'pointer',
                        transition: 'all 0.2s'
                      }}
                    >
                      {gender}
                    </button>
                  ))}
                </div>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px', marginBottom: '20px' }}>
                <div>
                  <label style={{ display: 'block', fontSize: '13px', fontWeight: '500', marginBottom: '8px' }}>
                    Weight (kg)
                  </label>
                  <input
                    type="number"
                    value={patientData.weight}
                    onChange={(e) => setPatientData(prev => ({ ...prev, weight: e.target.value }))}
                    placeholder="70"
                    style={{
                      width: '100%',
                      padding: '12px 16px',
                      backgroundColor: theme.bg,
                      border: `1px solid ${theme.border}`,
                      borderRadius: '8px',
                      color: theme.text,
                      fontSize: '14px',
                      boxSizing: 'border-box'
                    }}
                  />
                </div>
                <div>
                  <label style={{ display: 'block', fontSize: '13px', fontWeight: '500', marginBottom: '8px' }}>
                    Height (cm)
                  </label>
                  <input
                    type="number"
                    value={patientData.height}
                    onChange={(e) => setPatientData(prev => ({ ...prev, height: e.target.value }))}
                    placeholder="175"
                    style={{
                      width: '100%',
                      padding: '12px 16px',
                      backgroundColor: theme.bg,
                      border: `1px solid ${theme.border}`,
                      borderRadius: '8px',
                      color: theme.text,
                      fontSize: '14px',
                      boxSizing: 'border-box'
                    }}
                  />
                </div>
              </div>

              <div style={{ marginBottom: '20px' }}>
                <label style={{ display: 'block', fontSize: '13px', fontWeight: '500', marginBottom: '8px' }}>
                  Dominant Hand
                </label>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px' }}>
                  {['Right', 'Left'].map((hand) => (
                    <button
                      key={hand}
                      onClick={() => setPatientData(prev => ({ ...prev, writingHand: hand.toLowerCase() }))}
                      style={{
                        padding: '12px',
                        borderRadius: '8px',
                        border: 'none',
                        backgroundColor: patientData.writingHand === hand.toLowerCase() ? theme.primary : theme.bg,
                        color: patientData.writingHand === hand.toLowerCase() ? '#ffffff' : theme.text,
                        fontSize: '14px',
                        fontWeight: '500',
                        cursor: 'pointer',
                        transition: 'all 0.2s'
                      }}
                    >
                      {hand}
                    </button>
                  ))}
                </div>
              </div>

              <button
                onClick={() => setPatientData(prev => ({ ...prev, smoker: !prev.smoker }))}
                style={{
                  width: '100%',
                  display: 'flex',
                  alignItems: 'center',
                  padding: '16px',
                  backgroundColor: theme.bg,
                  border: 'none',
                  borderRadius: '8px',
                  cursor: 'pointer',
                  marginBottom: '20px'
                }}
              >
                <div style={{
                  width: '20px',
                  height: '20px',
                  borderRadius: '4px',
                  border: `2px solid ${patientData.smoker ? theme.primary : theme.border}`,
                  backgroundColor: patientData.smoker ? theme.primary : 'transparent',
                  marginRight: '12px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center'
                }}>
                  {patientData.smoker && <Check size={14} color="#ffffff" />}
                </div>
                <span style={{ fontSize: '14px', color: theme.text }}>
                  I am a smoker or have a history of smoking
                </span>
              </button>

              {bmi && (
                <div style={{
                  padding: '16px',
                  backgroundColor: `${theme.primary}15`,
                  border: `1px solid ${theme.primary}40`,
                  borderRadius: '8px'
                }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span style={{ fontSize: '13px', color: theme.textSecondary }}>Body Mass Index (BMI)</span>
                    <span style={{ fontSize: '18px', fontWeight: '600', color: theme.primary }}>{bmi}</span>
                  </div>
                </div>
              )}
            </div>

            <button
              onClick={() => setScreen('home')}
              disabled={!patientData.age || !patientData.gender}
              style={{
                width: '100%',
                padding: '16px',
                marginTop: '24px',
                borderRadius: '12px',
                border: 'none',
                backgroundColor: (patientData.age && patientData.gender) ? theme.primary : theme.border,
                color: (patientData.age && patientData.gender) ? '#ffffff' : theme.textSecondary,
                fontSize: '16px',
                fontWeight: '600',
                cursor: (patientData.age && patientData.gender) ? 'pointer' : 'not-allowed',
                transition: 'all 0.2s'
              }}
            >
              Save Information
            </button>
          </div>
        </div>
      </div>
    );
  }

  if (screen === 'results' && results) {
    const isPositive = results.result === 1;
    
    return (
      <div style={{ minHeight: '100vh', backgroundColor: theme.bg, color: theme.text }}>
        <div style={{ maxWidth: '480px', margin: '0 auto' }}>
          <Header title="Assessment Results" onBack={() => setScreen('home')} />

          <div style={{ padding: '24px' }}>
            <div style={{
              borderRadius: '16px',
              padding: '32px 24px',
              textAlign: 'center',
              backgroundColor: isPositive ? `${theme.danger}15` : `${theme.success}15`,
              border: `1px solid ${isPositive ? theme.danger : theme.success}40`,
              marginBottom: '24px'
            }}>
              <div style={{
                width: '80px',
                height: '80px',
                margin: '0 auto 20px',
                borderRadius: '50%',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                backgroundColor: isPositive ? `${theme.danger}20` : `${theme.success}20`
              }}>
                {isPositive ? (
                  <AlertTriangle size={40} color={theme.danger} />
                ) : (
                  <Check size={40} color={theme.success} />
                )}
              </div>
              <h3 style={{ fontSize: '20px', fontWeight: '700', marginBottom: '8px' }}>
                {isPositive ? 'Risk Indicators Detected' : 'Low Risk Assessment'}
              </h3>
              <p style={{ fontSize: '14px', color: theme.textSecondary, marginBottom: '16px' }}>
                Confidence Level
              </p>
              <div style={{ fontSize: '32px', fontWeight: '700', color: isPositive ? theme.danger : theme.success }}>
                {(results.confidence * 100).toFixed(1)}%
              </div>
            </div>

            <div style={{
              backgroundColor: theme.card,
              borderRadius: '16px',
              padding: '24px',
              marginBottom: '24px',
              border: `1px solid ${theme.border}`
            }}>
              <h3 style={{ fontSize: '16px', fontWeight: '600', marginBottom: '20px' }}>
                Assessment Breakdown
              </h3>
              
              <div style={{ marginBottom: '20px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                  <span style={{ fontSize: '13px', color: theme.textSecondary }}>Motor Control Analysis</span>
                  <span style={{ fontSize: '14px', fontWeight: '600', color: theme.text }}>
                    {(results.drawingScore * 100).toFixed(0)}%
                  </span>
                </div>
                <div style={{
                  width: '100%',
                  height: '8px',
                  backgroundColor: theme.bg,
                  borderRadius: '4px',
                  overflow: 'hidden'
                }}>
                  <div style={{
                    height: '100%',
                    width: `${results.drawingScore * 100}%`,
                    backgroundColor: theme.primary,
                    borderRadius: '4px',
                    transition: 'width 0.5s ease'
                  }} />
                </div>
              </div>

              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                  <span style={{ fontSize: '13px', color: theme.textSecondary }}>Clinical Risk Factors</span>
                  <span style={{ fontSize: '14px', fontWeight: '600', color: theme.text }}>
                    {(results.clinicalScore * 100).toFixed(0)}%
                  </span>
                </div>
                <div style={{
                  width: '100%',
                  height: '8px',
                  backgroundColor: theme.bg,
                  borderRadius: '4px',
                  overflow: 'hidden'
                }}>
                  <div style={{
                    height: '100%',
                    width: `${results.clinicalScore * 100}%`,
                    backgroundColor: theme.warning,
                    borderRadius: '4px',
                    transition: 'width 0.5s ease'
                  }} />
                </div>
              </div>
            </div>

            <div style={{
              backgroundColor: theme.card,
              borderRadius: '16px',
              padding: '20px',
              marginBottom: '24px',
              border: `1px solid ${theme.border}`
            }}>
              <h3 style={{ fontSize: '16px', fontWeight: '600', marginBottom: '12px' }}>
                Recommendations
              </h3>
              {isPositive ? (
                <ul style={{ margin: 0, paddingLeft: '20px', fontSize: '13px', lineHeight: '1.8', color: theme.textSecondary }}>
                  <li>Consult with a neurologist for comprehensive evaluation</li>
                  <li>Schedule appointment with primary care physician</li>
                  <li>Keep detailed symptom diary</li>
                  <li>Maintain regular physical activity</li>
                  <li>Follow-up screening in 6-12 months</li>
                </ul>
              ) : (
                <ul style={{ margin: 0, paddingLeft: '20px', fontSize: '13px', lineHeight: '1.8', color: theme.textSecondary }}>
                  <li>Continue regular health monitoring</li>
                  <li>Maintain active lifestyle</li>
                  <li>Follow up with primary care as needed</li>
                  <li>Consider repeat screening in 1-2 years</li>
                  <li>Report any new symptoms to healthcare provider</li>
                </ul>
              )}
            </div>

            <button
              onClick={() => generatePDF(results)}
              style={{
                width: '100%',
                padding: '16px',
                borderRadius: '12px',
                border: 'none',
                backgroundColor: theme.accent,
                color: '#ffffff',
                fontSize: '16px',
                fontWeight: '600',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                marginBottom: '12px'
              }}
            >
              <Download size={20} style={{ marginRight: '8px' }} />
              Download PDF Report
            </button>

            <div style={{
              padding: '16px',
              backgroundColor: `${theme.danger}15`,
              border: `1px solid ${theme.danger}40`,
              borderRadius: '12px',
              marginBottom: '24px'
            }}>
              <div style={{ display: 'flex', alignItems: 'flex-start' }}>
                <AlertTriangle size={18} color={theme.danger} style={{ marginRight: '12px', marginTop: '2px', flexShrink: 0 }} />
                <div style={{ fontSize: '12px', lineHeight: '1.5' }}>
                  <p style={{ fontWeight: '600', color: theme.danger, margin: '0 0 4px 0' }}>
                    Important Notice
                  </p>
                  <p style={{ color: theme.textSecondary, margin: 0 }}>
                    This is a screening assessment only, not a medical diagnosis. Please consult 
                    with a qualified healthcare professional for proper evaluation and diagnosis.
                  </p>
                </div>
              </div>
            </div>

            <button
              onClick={() => {
                setResults(null);
                setScreen('home');
              }}
              style={{
                width: '100%',
                padding: '16px',
                borderRadius: '12px',
                border: `1px solid ${theme.border}`,
                backgroundColor: theme.card,
                color: theme.text,
                fontSize: '16px',
                fontWeight: '600',
                cursor: 'pointer'
              }}
            >
              Complete Assessment
            </button>
          </div>
        </div>
      </div>
    );
  }

  if (screen === 'history') {
    return (
      <div style={{ minHeight: '100vh', backgroundColor: theme.bg, color: theme.text }}>
        <div style={{ maxWidth: '480px', margin: '0 auto' }}>
          <Header title="Assessment History" onBack={() => setScreen('home')} />

          <div style={{ padding: '24px' }}>
            {assessmentHistory.length === 0 ? (
              <div style={{ textAlign: 'center', padding: '60px 20px' }}>
                <BarChart3 size={48} color={theme.textSecondary} style={{ margin: '0 auto 16px' }} />
                <h3 style={{ fontSize: '18px', fontWeight: '600', marginBottom: '8px' }}>No Assessments Yet</h3>
                <p style={{ fontSize: '14px', color: theme.textSecondary, margin: 0 }}>
                  Complete an assessment to see your history here
                </p>
              </div>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                {assessmentHistory.map((assessment) => {
                  const date = new Date(assessment.date);
                  const isPositive = assessment.result === 1;
                  
                  return (
                    <button
                      key={assessment.id}
                      onClick={() => {
                        setSelectedReport(assessment);
                        setScreen('report-view');
                      }}
                      style={{
                        backgroundColor: theme.card,
                        border: `1px solid ${theme.border}`,
                        borderRadius: '12px',
                        padding: '16px',
                        textAlign: 'left',
                        cursor: 'pointer',
                        transition: 'all 0.2s'
                      }}
                    >
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '12px' }}>
                        <div>
                          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
                            <Calendar size={14} color={theme.textSecondary} />
                            <span style={{ fontSize: '13px', color: theme.textSecondary }}>
                              {date.toLocaleDateString()}
                            </span>
                          </div>
                          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                            <Clock size={14} color={theme.textSecondary} />
                            <span style={{ fontSize: '13px', color: theme.textSecondary }}>
                              {date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                            </span>
                          </div>
                        </div>
                        <span style={{
                          padding: '6px 12px',
                          borderRadius: '20px',
                          fontSize: '12px',
                          fontWeight: '500',
                          backgroundColor: isPositive ? `${theme.danger}20` : `${theme.success}20`,
                          color: isPositive ? theme.danger : theme.success
                        }}>
                          {isPositive ? 'Risk Detected' : 'Low Risk'}
                        </span>
                      </div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <span style={{ fontSize: '14px', color: theme.textSecondary }}>
                          Confidence: {(assessment.confidence * 100).toFixed(1)}%
                        </span>
                        <ChevronRight size={18} color={theme.textSecondary} />
                      </div>
                    </button>
                  );
                })}
              </div>
            )}
          </div>
        </div>
      </div>
    );
  }

  if (screen === 'report-view' && selectedReport) {
    const isPositive = selectedReport.result === 1;
    const date = new Date(selectedReport.date);
    
    return (
      <div style={{ minHeight: '100vh', backgroundColor: theme.bg, color: theme.text }}>
        <div style={{ maxWidth: '480px', margin: '0 auto' }}>
          <Header 
            title="Assessment Report" 
            onBack={() => setScreen('history')}
            rightAction={
              <button
                onClick={() => generatePDF(selectedReport)}
                style={{
                  background: 'none',
                  border: 'none',
                  color: theme.text,
                  cursor: 'pointer',
                  padding: '8px'
                }}
              >
                <Download size={20} />
              </button>
            }
          />

          <div style={{ padding: '24px' }}>
            <div style={{
              backgroundColor: theme.card,
              borderRadius: '16px',
              padding: '24px',
              border: `1px solid ${theme.border}`,
              marginBottom: '24px'
            }}>
              <h3 style={{ fontSize: '16px', fontWeight: '600', marginBottom: '16px' }}>Report Information</h3>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', fontSize: '14px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ color: theme.textSecondary }}>Report ID:</span>
                  <span style={{ fontWeight: '500', color: theme.text }}>{selectedReport.id}</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ color: theme.textSecondary }}>Date:</span>
                  <span style={{ fontWeight: '500', color: theme.text }}>{date.toLocaleDateString()}</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ color: theme.textSecondary }}>Time:</span>
                  <span style={{ fontWeight: '500', color: theme.text }}>
                    {date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                  </span>
                </div>
              </div>
            </div>

            {selectedReport.patientData && (
              <div style={{
                backgroundColor: theme.card,
                borderRadius: '16px',
                padding: '24px',
                border: `1px solid ${theme.border}`,
                marginBottom: '24px'
              }}>
                <h3 style={{ fontSize: '16px', fontWeight: '600', marginBottom: '16px' }}>Patient Information</h3>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', fontSize: '14px' }}>
                  {selectedReport.patientData.age && (
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <span style={{ color: theme.textSecondary }}>Age:</span>
                      <span style={{ fontWeight: '500', color: theme.text }}>
                        {selectedReport.patientData.age} years
                      </span>
                    </div>
                  )}
                  {selectedReport.patientData.gender && (
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <span style={{ color: theme.textSecondary }}>Gender:</span>
                      <span style={{ fontWeight: '500', color: theme.text }}>
                        {selectedReport.patientData.gender.charAt(0).toUpperCase() + selectedReport.patientData.gender.slice(1)}
                      </span>
                    </div>
                  )}
                  {selectedReport.patientData.weight && selectedReport.patientData.height && (
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <span style={{ color: theme.textSecondary }}>BMI:</span>
                      <span style={{ fontWeight: '500', color: theme.text }}>
                        {(selectedReport.patientData.weight / Math.pow(selectedReport.patientData.height / 100, 2)).toFixed(1)} kg/m²
                      </span>
                    </div>
                  )}
                </div>
              </div>
            )}

            <div style={{
              borderRadius: '16px',
              padding: '24px',
              textAlign: 'center',
              backgroundColor: isPositive ? `${theme.danger}15` : `${theme.success}15`,
              border: `1px solid ${isPositive ? theme.danger : theme.success}40`,
              marginBottom: '24px'
            }}>
              <div style={{
                width: '64px',
                height: '64px',
                margin: '0 auto 16px',
                borderRadius: '50%',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                backgroundColor: isPositive ? `${theme.danger}20` : `${theme.success}20`
              }}>
                {isPositive ? (
                  <AlertTriangle size={32} color={theme.danger} />
                ) : (
                  <Check size={32} color={theme.success} />
                )}
              </div>
              <h3 style={{ fontSize: '18px', fontWeight: '700', marginBottom: '8px' }}>
                {isPositive ? 'Risk Indicators Detected' : 'Low Risk Assessment'}
              </h3>
              <div style={{ fontSize: '28px', fontWeight: '700', color: isPositive ? theme.danger : theme.success }}>
                {(selectedReport.confidence * 100).toFixed(1)}%
              </div>
            </div>

            <button
              onClick={() => generatePDF(selectedReport)}
              style={{
                width: '100%',
                padding: '16px',
                borderRadius: '12px',
                border: 'none',
                backgroundColor: theme.primary,
                color: '#ffffff',
                fontSize: '16px',
                fontWeight: '600',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center'
              }}
            >
              <Download size={20} style={{ marginRight: '8px' }} />
              Download Full Report
            </button>
          </div>
        </div>
      </div>
    );
  }

  if (screen === 'about') {
    return (
      <div style={{ minHeight: '100vh', backgroundColor: theme.bg, color: theme.text }}>
        <div style={{ maxWidth: '480px', margin: '0 auto' }}>
          <Header title="About" onBack={() => setScreen('home')} />

          <div style={{ padding: '24px' }}>
            <div style={{ textAlign: 'center', marginBottom: '32px' }}>
              <div style={{
                width: '80px',
                height: '80px',
                backgroundColor: `${theme.primary}20`,
                borderRadius: '20px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                margin: '0 auto 16px',
                border: `2px solid ${theme.primary}`
              }}>
                <Activity size={40} color={theme.primary} />
              </div>
              <h2 style={{ fontSize: '24px', fontWeight: '700', marginBottom: '8px' }}>NeuroCheck</h2>
              <p style={{ color: theme.textSecondary, fontSize: '14px' }}>Version 1.0.0</p>
            </div>

            <div style={{
              backgroundColor: theme.card,
              borderRadius: '16px',
              padding: '24px',
              border: `1px solid ${theme.border}`,
              marginBottom: '24px'
            }}>
              <h3 style={{ fontSize: '16px', fontWeight: '600', marginBottom: '12px' }}>About This App</h3>
              <p style={{ fontSize: '14px', color: theme.textSecondary, lineHeight: '1.6', margin: 0 }}>
                NeuroCheck is an AI-powered screening tool designed to assist in early detection of Parkinson's disease. 
                The app combines motor assessment through drawing analysis with clinical risk factor evaluation to provide 
                a comprehensive screening result.
              </p>
            </div>

            <div style={{
              backgroundColor: theme.card,
              borderRadius: '16px',
              padding: '24px',
              border: `1px solid ${theme.border}`,
              marginBottom: '24px'
            }}>
              <h3 style={{ fontSize: '16px', fontWeight: '600', marginBottom: '16px' }}>Features</h3>
              <ul style={{ margin: 0, paddingLeft: '20px', fontSize: '14px', lineHeight: '1.8', color: theme.textSecondary }}>
                <li>Motor control assessment through pattern drawing</li>
                <li>Clinical risk factor analysis</li>
                <li>AI-powered prediction models</li>
                <li>Comprehensive PDF reports</li>
                <li>Assessment history tracking</li>
                <li>Dark mode support</li>
              </ul>
            </div>

            <div style={{
              backgroundColor: `${theme.danger}15`,
              border: `1px solid ${theme.danger}40`,
              borderRadius: '12px',
              padding: '16px'
            }}>
              <div style={{ display: 'flex', alignItems: 'flex-start' }}>
                <Info size={18} color={theme.danger} style={{ marginRight: '12px', marginTop: '2px', flexShrink: 0 }} />
                <div style={{ fontSize: '12px', lineHeight: '1.5' }}>
                  <p style={{ fontWeight: '600', color: theme.danger, margin: '0 0 8px 0' }}>
                    Medical Disclaimer
                  </p>
                  <p style={{ color: theme.textSecondary, margin: 0 }}>
                    This tool is for educational and research purposes only. It is NOT a substitute for professional 
                    medical diagnosis or treatment. Results should NOT be used as the sole basis for medical decisions. 
                    Consult qualified healthcare professionals for medical advice.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (isAnalyzing) {
    return (
      <div style={{
        minHeight: '100vh',
        backgroundColor: theme.bg,
        color: theme.text,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center'
      }}>
        <div style={{ textAlign: 'center' }}>
          <div style={{
            width: '64px',
            height: '64px',
            border: `4px solid ${theme.border}`,
            borderTopColor: theme.primary,
            borderRadius: '50%',
            animation: 'spin 1s linear infinite',
            margin: '0 auto 24px'
          }} />
          <h3 style={{ fontSize: '18px', fontWeight: '600', marginBottom: '8px' }}>
            Analyzing Assessment
          </h3>
          <p style={{ color: theme.textSecondary, fontSize: '14px' }}>
            Processing motor and clinical data...
          </p>
          <style>{`
            @keyframes spin {
              0% { transform: rotate(0deg); }
              100% { transform: rotate(360deg); }
            }
          `}</style>
        </div>
      </div>
    );
  }

  return null;
};

export default ParkinsonsApp;
