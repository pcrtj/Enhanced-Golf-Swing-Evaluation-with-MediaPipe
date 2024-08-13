
import { useState, useEffect } from "react";
import { Link, useNavigate } from 'react-router-dom';
import axios from "axios";
// import "../css/manage.css";
import '../index.css';
import '../css/newmanage.css';


function Works() {
  const navigate = useNavigate();
  const [people, setPeople] = useState([]);
  const [searchTerm, setSearchTerm] = useState("");

  useEffect(() => {
    const fetchPeople = async () => {
      try {
        const response = await axios.get("http://localhost:6956/peopleList");
        setPeople(response.data.peopleList); // สมมติว่า API ส่งค่ากลับมาในรูปแบบนี้
      } catch (error) {
        console.error("Error fetching data:", error);
      }
    };
    fetchPeople();
  }, []);
  const edit = (UID) => {
    sessionStorage.setItem("editID", UID)
    navigate("/editUser");
  }

  const handleSearchChange = (e) => {
    setSearchTerm(e.target.value);
  };

  const filteredPeople = searchTerm
    ? people.filter(person =>
      person.U_Firstname.toLowerCase().includes(searchTerm.toLowerCase()) ||
      person.U_Lastname.toLowerCase().includes(searchTerm.toLowerCase())
    )
    : people;

  return (
    <div>
      <div className="containerr flex flex-col items-center w-full">
        <div className="managebox">
          <div className="top-emotion">Students Manager</div>
            <div className="my-5 flex w-4/6 "> 
              <input
                type="search"
                className="searchbar w-full rounded-3xl mx-5 text-center shadow"
                placeholder="Enter student's name to search"
                onChange={handleSearchChange}
                required
              />
              <Link to="/addcrop" className="butaddmem">Add Student</Link>
            </div>
        </div>
        <div className="box-student">
          <ul id="List" className="list grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 p-4 ">
            {filteredPeople.map((person, index) => (
              <a onClick={() => edit(person.UID)} className="text-black">
                <li key={index} className="person-info flex flex-col bg-white rounded-lg shadow-lg p-4 space-y-2 ">
                  <div className="text-xl space-y-2">
                    <div>Firstname: {person.U_Firstname}</div>
                    <div>Lastname: {person.U_Lastname}</div>
                    <div>Gender: {person.U_Gender}</div>
                    <div>Birthday: {person.U_Birthday}</div>
                  </div>
                  <div className="flex justify-center items-center">
                    <img className="w-full h-64 object-cover rounded-md justify-self-center" src={`${person.U_PictureTitle}`} alt="profile" />
                  </div>
                </li>
              </a>

            ))}
          </ul>


        </div>
      </div>
    </div>

  );
}

export default Works;
