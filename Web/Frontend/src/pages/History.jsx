import axios from 'axios';
import { useEffect, useState } from 'react';
import '../css/history.css'

function History() {
  const [events, setEvents] = useState([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [modal, setModal] = useState(false);
  const [modalImgSrc, setModalImgSrc] = useState('');
  const fetchEvent = () => {
    axios.get('http://192.168.15.227:6956/getAllEvent')
      .then((response) => {
        setEvents(response.data);
        console.log("data is:", events)// อัปเดต state ด้วยข้อมูลจาก API
      })
      .catch((error) => {
        console.error('Error fetching events:', error);
      });
  };

  useEffect(() => {
    fetchEvent();
  }, []);

  // ฟังก์ชั่นการค้นหา
  const handleSearchChange = (e) => {
    setSearchTerm(e.target.value);
  };

  // กรอง events ตาม searchTerm
  const filteredEvents = events.filter(
    event =>
      event.U_Firstname.toLowerCase().includes(searchTerm.toLowerCase()) ||
      event.U_Lastname.toLowerCase().includes(searchTerm.toLowerCase())
  );
  const openModal = (imgSrc) => {
    setModalImgSrc(imgSrc);
    setModal(true);
  };

  // ฟังก์ชั่นสำหรับการปิด Modal
  const closeModal = () => {
    setModal(false);
  };

  return (
    <div>
      <div className="containerr flex flex-col items-center w-full">
        <div className="hisbox">
          <div className="top-emotion">Events history</div>
          <input
            type="text"
            className="searchhis"
            placeholder="Enter student's name to search"
            onChange={handleSearchChange}
          />
        </div>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 ml-5 mr-5">
        {filteredEvents.map((event, index) => (
          <div key={index} className="bg-white shadow-md rounded-lg p-4">
            <div className="font-bold text-lg mb-2">EID: {event.EID}</div>
            <div className="text-gray-700 mb-2">Name: {event.U_Firstname} {event.U_Lastname}</div>
            <div className="text-gray-700 mb-2">Emotion: {event.Emotion}</div>
            <div className="text-gray-700 mb-2">Gender: {event.Gender}</div>
            <div className="text-gray-700 mb-2">Age: {event.Age}</div>
            <div className="text-gray-700 mb-2">
              Time: {
                new Date(event.EDateTime).toLocaleDateString('en-US', {
                  year: 'numeric',
                  month: 'long',
                  day: 'numeric',
                  weekday: 'long',
                })} at {
                new Date(event.EDateTime).toLocaleTimeString('en-US', {
                  hour: '2-digit',
                  minute: '2-digit',
                  second: '2-digit',
                  hour12: true
                })
              }
            </div>
            <div className="mb-2">
              FacePicture: <img
                className="h-48 object-cover rounded-md mb-2"
                src={`data:image/png;base64,${event.FacePicture}`}
                alt="Face"
                onClick={() => openModal(`data:image/png;base64,${event.FacePicture}`)} // เปลี่ยนจาก onMouseEnter เป็น onClick
              />
            </div>

            <div className="mb-2">
              EnvironmentPicture: <img className="w-full h-48 object-cover rounded-md mb-2" src={`data:image/png;base64,${event.EnvironmentPicture}`} alt="Environment" onClick={() => openModal(`data:image/png;base64,${event.EnvironmentPicture}`)} />
            </div>
          </div>
        ))}
      </div>
      {modal && (
        <div className="modal" onClick={closeModal}>
          <img className="modal-content modal-img" src={modalImgSrc} alt="Modal" />
        </div>
      )}
    </div>
  );
}

export default History;