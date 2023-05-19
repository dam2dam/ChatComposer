import axios from "axios";
import React, { useEffect, useState } from "react";

function PianoChat({ setNotes }) {
  const ChatApi = axios.create();
  const [message, setMessage] = useState("");
  const [input, setInput] = useState("");

  const [baseUrl, setBaseUrl] = useState("https://k8a504.p.ssafy.io");

  useEffect(() => {
    if (window.location.host.startsWith("localhost")) {
      setBaseUrl("http://localhost:3000");
    }
  }, []);

  return (
    <div>
      <input value={input} onChange={(e) => setInput(e.target.value)}></input>
      <button
        onClick={() => {
          ChatApi.post(`${baseUrl}/api/chatgpt`, {
            message: input,
            // prevData,
          }).then((res) => {
            setMessage(JSON.stringify(res.data));
            setNotes(res.data.noteInfo);
          });
        }}
      >
        전송하기
      </button>
      {/* <Link href="/sequencer">악보로 이동하기</Link> */}
    </div>
  );
}

export default PianoChat;
