import React, { useState, useEffect } from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';
import '../css/History.css';

const CircularProgress = ({ value }) => {
  const radius = 30;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference - (value / 100) * circumference;

  const getColor = (value) => {
    if (value < 31) return '#FF0000';
    if (value < 61) return '#FFFF00';
    return '#00FF00';
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
  document.body.style.overflow = "hidden";

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
      console.log('Fetched history data:', data); // For debugging
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

  const getBarColor = (value) => {
    if (value < 31) return '#FF0000';
    if (value < 61) return '#FFFF00';
    return '#00FF00';
  };

  const renderTableRows = () => {
    if (historyData.length === 0) {
      return (
        <tr>
          <td colSpan="5" className='nodata'>No history data available.</td>
        </tr>
      );
    }
    return historyData.map((item) => (
      <tr key={item.E_ID}>
        <td>{item.date}</td>
        <td>
          <video width="200" controls>
            <source src={item.inputClip} type="video/mp4" />
            Your browser does not support the video tag.
          </video>
        </td>
        <td>
          <video width="200" controls>
            <source src={item.outputClip} type="video/mp4" />
            Your browser does not support the video tag.
          </video>
        </td>
        <td>
          <div className="chart-container">
            <ResponsiveContainer width="100%" height={100}>
              <BarChart data={item.accuracy.map((value, idx) => ({ name: swingPhases[idx], value }))}>
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
                  contentStyle={{ backgroundColor: '#1a4731', border: 'none', borderRadius: '4px' }}
                  cursor={{ fill: 'transparent' }}
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