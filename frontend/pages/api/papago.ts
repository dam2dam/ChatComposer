import type { NextApiRequest, NextApiResponse } from "next";
import axios from "axios";

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse,
) {
  const body = req.body;

  let englishMessage = body.message;
  //주석 console.log("파파고번역");
  //주석 console.log(englishMessage);
  //만일 현재 데이터에 한국어가 포함되어 있다면, 파파고를 호출하여 영어로 번역합니다 결과를 반환합니다.
  if (/[ㄱ-ㅎ|ㅏ-ㅣ|가-힣]/.test(englishMessage)) {
    //파파고 api 호출
    const papagoApi = axios.create({
      baseURL: "https://openapi.naver.com/v1/papago/n2mt",
      headers: {
        "X-Naver-Client-Id": process.env.X_NAVER_CLIENT_ID,
        "X-Naver-Client-Secret": process.env.X_NAVER_SECRET_ID,
      },
    });
    //파파고 한-영 번역을 거친 결과를 englishMessage로 저장
    englishMessage = await papagoApi
      .post("", { source: "ko", target: "en", text: englishMessage })
      .then((res) => res.data.message.result.translatedText);
    // .catch((err) => //주석 console.log(JSON.stringify(err)));
  }

  //주석 console.log(englishMessage);
  res.status(200).json({
    message: englishMessage,
  });
}
