import axios from 'axios';


// 태그 생성

interface TagRequest {
  tag: string;
}

interface TagResponse {
  msg: string;
}

export async function createTag(tag: string): Promise<TagResponse> {
  const url = '/tags';
  const data: TagRequest = { tag };
  try {
    const response = await axios.post<TagResponse>(url, data);
    return response.data;
  } catch (error) {
    if (error.response) {
      throw new Error('태그 생성에 실패했습니다.'); // 서버에서 에러 응답을 보낼 경우
    } else {
      throw new Error('서버와의 연결이 끊어졌습니다.'); // 서버와의 연결이 끊어졌거나 요청 자체가 실패한 경우
    }
  }
}





// 태그 목록 조회

interface Tag {
  tag_id: number;
  tag_name: string;
}

interface GetTagsResponse {
  tags: Tag[];
}

async function getTags(): Promise<Tag[]> {
  const url = '/tags';
  try {
    const response = await axios.get<GetTagsResponse>(url);
    return response.data.tags;
  } catch (error) {
    if (error.response) {
      throw new Error('태그 목록 조회에 실패했습니다.');
    } else {
      throw new Error('서버와의 연결이 끊어졌습니다.');
    }
  }
}