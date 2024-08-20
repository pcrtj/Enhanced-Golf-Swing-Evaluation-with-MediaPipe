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
          GOLF <span></span>ANALYSIS
        </Link>
        <nav className={`${"header__content__nav"} ${menuOpen && size.width < 768 ? `${"isMenu"}` : ""} }`}>
          <ul>
            <li>
              <Link to="/home">
                <button className="loginlogout">
                  Home
                </button>
              </Link>
            </li>
            <li>
              <Link to="/history">
                <button className="loginlogout">
                  History
                </button>
              </Link>
            </li>
            {/* <li>
              <Link to="/login">
                <button className="loginlogout" onClick={logout}>Logout</button>
              </Link>
            </li> */}
            <li>
              <Link to="/login">
                <button className="Btn">
                  <div className="sign"><svg viewBox="0 0 512 512"><path d="M377.9 105.9L500.7 228.7c7.2 7.2 11.3 17.1 11.3 27.3s-4.1 20.1-11.3 27.3L377.9 406.1c-6.4 6.4-15 9.9-24 9.9c-18.7 0-33.9-15.2-33.9-33.9l0-62.1-128 0c-17.7 0-32-14.3-32-32l0-64c0-17.7 14.3-32 32-32l128 0 0-62.1c0-18.7 15.2-33.9 33.9-33.9c9 0 17.6 3.6 24 9.9zM160 96L96 96c-17.7 0-32 14.3-32 32l0 256c0 17.7 14.3 32 32 32l64 0c17.7 0 32 14.3 32 32s-14.3 32-32 32l-64 0c-53 0-96-43-96-96L0 128C0 75 43 32 96 32l64 0c17.7 0 32 14.3 32 32s-14.3 32-32 32z"></path></svg></div>
                  <div className="text" onClick={logout} >Logout</div>
                </button>
              </Link>
            </li>
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
