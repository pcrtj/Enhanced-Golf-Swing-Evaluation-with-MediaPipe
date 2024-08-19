import { useEffect, useState } from "react";
import { BiMenuAltRight } from "react-icons/bi";
import { AiOutlineClose } from "react-icons/ai";
import { Link} from "react-router-dom";
import '../css/Navbar.css'

function Navbar() {
  const [menuOpen, setMenuOpen] = useState(false);
  const [size, setSize] = useState({
    width: 0,
    height: 0,
  });
  useEffect(() => {
    const handleResize = () => {
      setSize({
        width: window.innerWidth,
        height: window.innerHeight,
      });
    };
    window.addEventListener("resize", handleResize);

    return () => window.removeEventListener("resize", handleResize);
  }, []);

  useEffect(() => {
    if (size.width > 768 && menuOpen) {
      setMenuOpen(false);
    }
  }, [size.width, menuOpen]);

  const menuToggleHandler = () => {
    setMenuOpen((p) => !p);
  };

  const logout=()=>{
    sessionStorage.setItem("username",'null')
  }

  return (
    <header className="header">
      <div className="header__content">
        <Link to="/home" className="header__content__logo">
          Welcome Back
        </Link>
        <nav className={`${"header__content__nav"} ${menuOpen && size.width < 768 ? `${"isMenu"}` : ""} }`}>
          <ul>
            <li>
                <Link to="/home"activeClassName="active-home" className="home">Home</Link>
            </li>
            <li>
              <Link to="/history"className="search">History</Link>
            </li>
            <li>
              <Link to="/manage"className="manage">Students</Link>
            </li>
            <li>
              <Link to="/emotions"className="manage">Emotions</Link>
            </li>
            
            <Link to="/login">
              <button className="btn btn__login" onClick={logout}>Logout</button>
            </Link>
          </ul>
        </nav>
        <div className="header__content__toggle">
          {!menuOpen ? (
            <BiMenuAltRight onClick={menuToggleHandler} />
          ) : (
            <AiOutlineClose onClick={menuToggleHandler} />
          )}
        </div>
      </div>
    </header>
  );
}

export default Navbar;