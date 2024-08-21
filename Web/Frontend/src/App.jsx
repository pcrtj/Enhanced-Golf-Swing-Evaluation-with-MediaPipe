import './App.css';
import { HashRouter, Routes, Route } from "react-router-dom";
import Navbar from "./pages/Navbar";
import Navbarguest from "./pages/Navbarguest";
import Homeuser from "./pages/Homeuser";
import Homeguest from "./pages/Homeguest";
import Login from "./pages/Login";
import History from "./pages/History"

function App() {
  return (
    <div className='maindisplay'>
      <HashRouter>
        <Routes>
          {/* เส้นทางเข้าสู่ระบบ */}
          <Route path="/login" element={<Login />} />
          
          {/* เส้นทางสำหรับ guest */}
          <Route 
            path="/" 
            element={
              <>
                <Navbarguest />
                <Homeguest />
              </>
            } 
          />
          
          {/* เส้นทางสำหรับ user */}
          <Route 
            path="/homeuser" 
            element={
              <>
                <Navbar />
                <Homeuser />
              </>
            } 
          />

          <Route
            path="/history"
            element={
              <>
                <Navbar />
                <History />
              </>
            }
            />
        </Routes>
      </HashRouter>
    </div>
  );
}

export default App;
