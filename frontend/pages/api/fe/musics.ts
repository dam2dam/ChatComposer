import fs from "fs";
import { promisify } from "util";

const readFile = promisify(fs.readFile);
const writeFile = promisify(fs.writeFile);

export default async function handler(req, res) {
  if (req.method === "POST") {
    try {
      const data = req.body;

      // 기존 데이터 불러오기
      const jsonData = await readFile("data.json", "utf-8");
      const existingData = JSON.parse(jsonData);

      // 마지막 pk 값 구하기
      const lastPk =
        existingData.length > 0 ? existingData[existingData.length - 1].pk : 0;

      // 새로운 데이터에 pk 추가하기
      const newData = { ...data, pk: lastPk + 1 };

      // 기존 데이터와 새로운 데이터 병합
      const mergedData = [...existingData, newData];

      // 병합된 데이터를 JSON 파일로 저장
      await writeFile("data.json", JSON.stringify(mergedData));

      // 저장된 데이터와 함께 성공 응답
      res.status(200).json({ success: true, data: newData });
    } catch (error) {
      console.error(error);
      res.status(500).json({ success: false, message: "Failed to save data." });
    }
  } else {
    res
      .status(404)
      .json({ success: false, message: "Invalid request method." });
  }
}
