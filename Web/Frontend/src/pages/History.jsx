import React, { useState, useEffect, useRef } from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';
import '../css/History.css';

const CircularProgress = ({ value }) => {
  const radius = 30;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference - (value / 100) * circumference;

  const getColor = (value) => {
    if (value < 41) return '#EB4343';
    if (value < 51) return '#FF9933';
    if (value < 61) return '#FFFF33';
    return '#99FF00';
  };

  return (
    <div className="circular-progress">
      <svg width="80" height="80">
        <circle
          stroke="#2d6a4f"
          strokeWidth="5"
          fill="transparent"
          r={radius}
          cx="40"
          cy="40"
        />
        <circle
          stroke={getColor(value)}
          strokeWidth="5"
          strokeDasharray={circumference}
          strokeDashoffset={strokeDashoffset}
          strokeLinecap="round"
          fill="transparent"
          r={radius}
          cx="40"
          cy="40"
        />
      </svg>
      <span className="circular-progress-value">{`${Math.round(value)}%`}</span>
    </div>
  );
};

const HistoryTable = () => {
  // document.body.style.overflow = "hidden";

  const [currentPage, setCurrentPage] = useState(1);
  const [historyData, setHistoryData] = useState([]);
  const [username, setUsername] = useState('');
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const rowsPerPage = 4;

  useEffect(() => {
    const storedUsername = sessionStorage.getItem('username');
    if (storedUsername) {
      setUsername(storedUsername);
    } else {
      setError('Username not found. Please log in.');
    }
  }, []);

  useEffect(() => {
    if (username) {
      fetchHistoryData();
    }
  }, [currentPage, username]);

  const fetchHistoryData = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`http://localhost:3000/history?username=${username}&page=${currentPage}`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const contentType = response.headers.get("content-type");
      if (!contentType || !contentType.includes("application/json")) {
        throw new Error("Oops! We haven't received a valid JSON response from the server.");
      }
      const data = await response.json();
      console.log('Fetched history data:', data);
      setHistoryData(data);
    } catch (error) {
      console.error('Error fetching history data:', error);
      setError(`Failed to load history data: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  const nextPage = () => {
    setCurrentPage(currentPage + 1);
  };

  const prevPage = () => {
    if (currentPage > 1) {
      setCurrentPage(currentPage - 1);
    }
  };

  const swingPhases = ['A', 'TU', 'MB', 'T', 'MD', 'I', 'MFT', 'F'];
  const swingPhasesFullNames = [
    'Address', 'Toe Up', 'Mid Backswing', 'Top',
    'Mid Downswing', 'Impact', 'Mid Follow Through', 'Finish'
  ];
  const angleNames = [
    'Left Shoulder', 'Right Shoulder', 'Left Elbow', 'Right Elbow',
    'Left Hip', 'Right Hip', 'Left Knee', 'Right Knee'
  ];

  const getBarColor = (value) => {
    if (value < 41) return '#EB4343';
    if (value < 51) return '#FF9933';
    if (value < 61) return '#FFFF33';
    return '#99FF00';
  };

  const calculateAverageAccuracy = (accuracies) => {
    return accuracies.reduce((sum, acc) => sum + acc, 0) / accuracies.length;
  };

  const CustomTooltip = ({ active, payload, label, rowIndex }) => {
    const tooltipRef = useRef(null);
  
    useEffect(() => {
      if (tooltipRef.current && active) {
        const tooltipElement = tooltipRef.current;
        
        if (rowIndex === 2) {
          tooltipElement.style.bottom = 'auto';
          tooltipElement.style.top = '-180px';
        } else {
          tooltipElement.style.top = '-120px';
          tooltipElement.style.bottom = 'auto';
        }
      }
    }, [active, label, rowIndex]);
  
    const getBackgroundColor = (value) => {
      if (value < 41) return 'rgba(220, 53, 69, 0.8)';
      if (value < 51) return 'rgba(255, 120, 7, 0.8)';
      if (value < 61) return 'rgba(255, 244, 0, 0.6)';
      return 'rgba(40, 167, 69, 0.8)';
    };
  
    const getTextColor = (value) => {
      return '#ffffff';  // ใช้สีขาวสำหรับข้อความทั้งหมด
    };
  
    if (active && payload && payload.length) {
      const angleData = payload[0].payload.angleData;
      const fullName = swingPhasesFullNames[swingPhases.indexOf(label)];
      return (
        <div ref={tooltipRef} className="custom-tooltip" style={{ 
          backgroundColor: 'rgba(52, 58, 64, 0.9)', 
          color: '#ffffff', 
          padding: '12px', 
          borderRadius: '8px',
          zIndex: 1000,
          position: 'relative',
          width: '450px',
          marginTop: '50px',
          boxShadow: '0 4px 6px rgba(0,0,0,0.1)'
        }}>
          <p style={{ margin: '0 0 8px 0', fontWeight: 'bold', borderBottom: '1px solid rgba(255,255,255,0.2)', paddingBottom: '8px' }}>
            <span style={{float: 'left'}}>{fullName}:</span>
            <span style={{float: 'right'}}>{payload[0].value.toFixed(2)}%</span>
            <div style={{clear: 'both'}}></div>
          </p>
          <p style={{ margin: '8px 0', fontWeight: 'bold' }}>Joint Performance:</p>
          <div style={{ 
            display: 'grid', 
            gridTemplateColumns: '1fr 1fr', 
            gap: '8px'
          }}>
            {[0, 2, 4, 6].map((startIndex) => (
              <React.Fragment key={startIndex}>
                {[startIndex, startIndex + 1].map((index) => {
                  const value = angleData[index];
                  return (
                    <div key={angleNames[index]} style={{
                      backgroundColor: getBackgroundColor(value),
                      color: getTextColor(value),
                      padding: '4px 8px',
                      borderRadius: '4px',
                      fontWeight: 700,
                      display: 'flex',
                      justifyContent: 'space-between',
                      alignItems: 'center'
                    }}>
                      <span>{angleNames[index]}:</span>
                      <span>{value.toFixed(2)}%</span>
                    </div>
                  );
                })}
              </React.Fragment>
            ))}
          </div>
        </div>
      );
    }
    return null;
  };

  const renderTableRows = () => {
    if (historyData.length === 0) {
      return (
        <tr>
          <td colSpan="5" className='nodata'>No history data available.</td>
        </tr>
      );
    }
    return historyData.slice(0, rowsPerPage).map((item, rowIndex) => (
      <tr key={item.E_ID}>
        <td>{item.date}</td>
        <td>
          <video width="200" height="250" controls>
            <source src={`http://localhost:3000/uploads/${item.inputClip}`} type="video/mp4" />
            Your browser does not support the video tag.
          </video>
        </td>
        <td>
          <video width="200" height="250" controls>
            <source src={`http://localhost:3000/uploads/${item.outputClip}`} type="video/mp4" />
            Your browser does not support the video tag.
          </video>
        </td>
        <td>
          <div className="chart-container" style={{ position: 'relative', height: '100px' }}>
            <ResponsiveContainer width="100%" height={100}>
              <BarChart
                data={item.accuracy.map((angleAccuracies, idx) => ({
                  name: swingPhases[idx],
                  fullName: swingPhasesFullNames[idx],
                  value: calculateAverageAccuracy(angleAccuracies),
                  angleData: angleAccuracies
                }))}
                margin={{ top: 20, right: 0, left: 0, bottom: 0 }}
              >
                <XAxis
                  dataKey="name"
                  stroke="#ffffff"
                  axisLine={{ stroke: '#ffffff' }}
                  tickLine={{ stroke: '#ffffff' }}
                />
                <YAxis
                  stroke="#ffffff"
                  axisLine={{ stroke: '#ffffff' }}
                  tickLine={{ stroke: '#ffffff' }}
                  domain={[0, 100]}
                />
                <Tooltip
                  content={<CustomTooltip rowIndex={rowIndex} />}
                  cursor={{ fill: 'transparent' }}
                  position={{ y: 0 }}
                  wrapperStyle={{ zIndex: 1000, visibility: 'visible' }}
                />
                <Bar
                  dataKey="value"
                  fill="#FFFFFF"
                  shape={(props) => {
                    const { x, y, width, height, value } = props;
                    return (
                      <rect
                        x={x}
                        y={y}
                        width={width}
                        height={height}
                        fill={getBarColor(value)}
                        stroke="none"
                      />
                    );
                  }}
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </td>
        <td>
          <CircularProgress value={item.avgAccuracy} />
        </td>
      </tr>
    ));
  };

  if (loading) {
    return <div className="loading-message">Loading history data...</div>;
  }

  if (error) {
    return <div className="error-message">{error}</div>;
  }

  return (
    <div className="history-container">
      {username ? (
        <>
          <div className="table-wrapper">
            <table className="history-table">
              <thead>
                <tr>
                  <th>Date/Time</th>
                  <th>Input Clip</th>
                  <th>Output Clip</th>
                  <th>Accuracy per Phase</th>
                  <th>Average Accuracy</th>
                </tr>
              </thead>
              <tbody>
                {renderTableRows()}
              </tbody>
            </table>
          </div>
          <div className="pagination">
            <button
              onClick={prevPage}
              disabled={currentPage === 1}
              className="pagination-button-pre"
            >
              Previous
            </button>
            <button
              onClick={nextPage}
              disabled={historyData.length < rowsPerPage}
              className="pagination-button-next"
            >
              Next
            </button>
          </div>
        </>
      ) : (
        <div className="login-message">Please log in to view your history.</div>
      )}
    </div>
  );
};

export default HistoryTable;