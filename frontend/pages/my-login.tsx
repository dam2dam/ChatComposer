import Link from "next/link";
import React from "react";
import serverApi from "../services/serverApi";
import axios from "axios";

function Login() {
  // const url = `/oauth2/authorization/google?redirect_uri=http://localhost:3000/oauth/redirect`;
  const url = `/oauth2/authorization/google?redirect_uri=https://k8a504.p.ssafy.io/oauth/redirect`;
  // const url = `/oauth2/authorization/google`;

  const onClick = () => {
    // axios.get(`/api/login?url=${url}`);
    serverApi
      .get(url)
      .then((res) => {
        //주석 console.log(res);
        return res;
      })
      .catch((err) => {
        //주석 console.log("로그인버튼에서나는에러");
        //주석 console.log(err);
      });
  };
  return (
    <div>
      <div className="bg-slate-500" onClick={onClick} role="button">
        로그인하기
      </div>
      <a
        className="bg-red-400"
        href="https://k8a504.p.ssafy.io:8080/api/oauth2/authorization/google?redirect_uri=https://k8a504.p.ssafy.io/"
      >
        로그인링크
      </a>
      <a
        className="bg-red-400"
        href="https://k8a504.p.ssafy.io:8080/api/oauth2/authorization/google?redirect_uri=https://k8a504.p.ssafy.io:3000/oauth/redirect
        "
      >
        로그인링크
      </a>
    </div>
  );
}

export default Login;
