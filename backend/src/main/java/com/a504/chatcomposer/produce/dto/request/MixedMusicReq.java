package com.a504.chatcomposer.produce.dto.request;

import com.a504.chatcomposer.produce.dto.response.FileResp;
import lombok.*;
import org.springframework.web.multipart.MultipartFile;

/**
 * 믹싱 음악을 생성하기 위한 API ([POST] /produce/musics/riffusion) 요청에 필요한 리퀘스트 바디 정의.
 */
@Data
@NoArgsConstructor(access = AccessLevel.PROTECTED)
@AllArgsConstructor
@Builder
public class MixedMusicReq {
    private String musicSource;
    private String prompt;

    public static MixedMusicReq of(String musicSource, String prompt) {
        MixedMusicReq req = new MixedMusicReq();
        req.setMusicSource(musicSource);
        req.setPrompt(prompt);
        return req;
    }
}