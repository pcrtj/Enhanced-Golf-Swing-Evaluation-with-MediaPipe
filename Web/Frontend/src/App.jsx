import './App.css'
import { HashRouter, Routes, Route } from "React-router-dom";
import Navbar from "./pages/Navbar";
import Home from "./pages/Home";
import Footer from "./pages/Footer";
import Login from "./pages/Login";
import Register from "./pages/Register";

function App() {
  return (
    <div className='maindisplay'>
      <HashRouter>
        <Routes>
          {/* Specify routes without Navbar */}
          <Route path="/login/*" element={<Login />} />
          <Route path="/" element={<Login />} />
          <Route path="/register" element={<Register />} />

          {/* Specify routes with Navbar */}
          <Route path="/*" element={
              <>
                <Navbar />
                <Routes>
                  <Route path="/home" element={<Home />} />
                </Routes>
                <Footer />
              </>
            }
          />
        </Routes>
        
      </HashRouter>
    </div>
  );
}

export default App;
