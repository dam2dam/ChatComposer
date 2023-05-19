import Link from 'next/link'
import React, { useEffect, useState } from 'react'


export default function Signup() {
  const [email, setEmail] = useState("");

  const [errorMap, setErrorMap] = useState({
    emailError: "",
    nicknameError: "",
    passwordError: "",
  });

  const [errorMessage, setErrorMessage] = useState("");
  useEffect(() => {
    let newMessages = "";
    Object.values(errorMap).forEach((e) => {
      if (e) newMessages = e;
    });
    setErrorMessage(newMessages);
  }, [errorMap]);

  const [nickname, setNickname] = useState("");
    //닉네임무결성
    useEffect(() => {
      if (nickname.length) {
        if (nickname.length > 14 || nickname.length < 2) {
          setErrorMap((prev) => ({
            ...prev,
            nicknameError: "닉네임은 2~14글자 입니다.",
          }));
        } else if (
          !nickname.match(
            /^(?=.*[a-zA-Z0-9ㄱ-ㅎ가-힣])[a-zA-Z0-9ㄱ-ㅎ가-힣]{2,14}$/,
          )
        ) {
          setErrorMap((prev) => ({
            ...prev,
            nicknameError: "닉네임에 특수문자를 사용할 수 없습니다.",
          }));
        } else {
          setErrorMap((prev) => ({
            ...prev,
            nicknameError: "",
          }));
        }
      }
    }, [nickname]);
    
  const [password, setPassword] = useState("")

  const [passwordConfirm, setPasswordConfirm] = useState("")

  const [genre, setGenre] = useState("")

  return(  
    <div>
      <div className="bg-grey-lighter min-h-screen flex flex-col">
        <div className="container max-w-sm mx-auto flex-1 flex flex-col items-center justify-center px-2">
          <div className="bg-white px-6 py-8 rounded shadow-md text-black w-full">
            <h1 className="mb-8 text-3xl text-center">Sign up</h1>
            <input
              className="block border border-grey-light w-full p-3 rounded mb-4"
              type="text"
              name="email"
              placeholder="Email" 
              value={email}
              onChange={(e) => {
                setEmail(e.target.value)
              }}
            />
            <input
              className="block border border-grey-light w-full p-3 rounded mb-4"
              type="text"
              name="nickname"
              placeholder="Nickname" 
              value={nickname}
              onChange={(e) => {
                setNickname(e.target.value)
              }}
            />
            <input
              type="password"
              className="block border border-grey-light w-full p-3 rounded mb-4"
              name="password"
              placeholder="Password" 
              value={password}
              onChange={(e) => {
                setPassword(e.target.value);
              }}
            />
            <input
              type="password"
              className="block border border-grey-light w-full p-3 rounded mb-4"
              name="confirm_password"
              placeholder="Confirm Password" 
              value={passwordConfirm}
              onChange={(e) => {
                setPasswordConfirm(e.target.value);
              }}
            />

            <div className="w-full relative inline-flex mb-4">
              <svg className="w-2 h-2 absolute top-0 right-0 m-4 pointer-events-none" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 412 232"><path d="M206 171.144L42.678 7.822c-9.763-9.763-25.592-9.763-35.355 0-9.763 9.764-9.763 25.592 0 35.355l181 181c4.88 4.882 11.279 7.323 17.677 7.323s12.796-2.441 17.678-7.322l181-181c9.763-9.764 9.763-25.592 0-35.355-9.763-9.763-25.592-9.763-35.355 0L206 171.144z" fill="#648299" fill-rule="nonzero"/></svg>
              <select className="w-full border border-gray-300 rounded-full text-gray-600 h-10 pl-5 pr-10 bg-white hover:border-gray-400 focus:outline-none appearance-none">
                <option>Genre you prefer</option>
                <option>Hiphop</option>
                <option>Classic</option>
                <option>Jazz</option>
                <option>Etc</option>
              </select>
            </div>

            <button
              type="submit"
              className="w-full text-center py-3 rounded bg-blue-500 text-white hover:bg-green-dark focus:outline-none"
            >Create Account</button>
          </div>

          <div className="text-grey-dark mt-6">
            Already have an account?
            <Link className="no-underline border-b border-blue text-blue" href="../login/">
              Log in
            </Link>
          </div>
        </div>
      </div>
    </div>
  )
}