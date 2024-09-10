import { useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';
import { mockHistoryData } from '../mockupdata/historyData';
import '../css/History.css';

const CircularProgress = ({ value }) => {
  const radius = 30;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference - (value / 100) * circumference;

  const getColor = (value) => {
    if (value <= 30) return '#FF0000';
    if (value <= 60) return '#FFFF00';
    return '#00FF00';
  };

  return (
    <div className="circular-progress">
      <svg>
        <circle
          stroke="#ccc"
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
  const rowsPerPage = 4;

  const historyData = mockHistoryData;

  const startIdx = (currentPage - 1) * rowsPerPage;
  const endIdx = Math.min(startIdx + rowsPerPage, historyData.length);

  const nextPage = () => {
    if (currentPage * rowsPerPage < historyData.length) {
      setCurrentPage(currentPage + 1);
    }
  };

  const prevPage = () => {
    if (currentPage > 1) {
      setCurrentPage(currentPage - 1);
    }
  };

  const getBarColor = (value) => {
    if (value <= 30) return '#3498DB';  // เปลี่ยนเป็นสีฟ้า
    if (value <= 60) return '#E67E22';  // เปลี่ยนเป็นสีส้ม
    return '#2ECC71';                   // เปลี่ยนเป็นสีเขียวเข้ม
  };

  const swingPhases = ['A', 'TU', 'MB', 'T', 'MD', 'I', 'MFT', 'F'];

  const renderTableRows = () => {
    const rows = [];
    for (let i = 0; i < rowsPerPage; i++) {
      const item = historyData[startIdx + i];
      if (item) {
        rows.push(
          <tr key={item.id}>
            <td>{item.date}</td>
            <td>{item.inputClip}</td>
            <td>{item.outputClip}</td>
            <td>
              <div className="chart-container">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={item.accuracy.map((value, idx) => ({ name: swingPhases[idx], value }))}>
                    <XAxis dataKey="name" stroke="#ffffff" />
                    <YAxis stroke="#ffffff" />
                    <Tooltip contentStyle={{ backgroundColor: '#1a4731', border: 'none', borderRadius: '4px' }} />
                    <Bar dataKey="value" fill={(entry) => getBarColor(entry.value)} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </td>
            <td>
              <CircularProgress value={item.avgAccuracy} />
            </td>
          </tr>
        );
      } else {
        rows.push(<tr key={`empty-${i}`}><td colSpan="5">&nbsp;</td></tr>);
      }
    }
    return rows;
  };

  return (
    <div className="history-container">
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
          className="pagination-button"
        >
          Previous
        </button>
        <button
          onClick={nextPage}
          disabled={endIdx >= historyData.length}
          className="pagination-button"
        >
          Next
        </button>
      </div>
    </div>
  );
};

export default HistoryTable;
