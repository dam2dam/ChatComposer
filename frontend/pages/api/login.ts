import { NextApiRequest, NextApiResponse } from "next";

export default function handler(req: NextApiRequest, res: NextApiResponse) {
  const { url } = req.query; // 요청에서 url 매개변수를 추출합니다.

  // url을 사용하여 필요한 작업을 수행합니다.

  res.status(200).json({ message: "Success" });
}
