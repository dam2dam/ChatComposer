package com.a504.chatcomposer.produce.service;

import com.a504.chatcomposer.produce.dto.request.MultipartFileReq;
import com.a504.chatcomposer.produce.dto.request.OriginalMusicReq;
import com.a504.chatcomposer.produce.dto.response.FileUrlResp;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;

/**
 * Produce 관련 비즈니스 로직 처리를 위한 서비스 인터페이스 정의.
 */
public interface ProduceService {
    // 믹싱 음악 생성
    byte[] createMusic(String musicSource, String prompt) throws IOException, InterruptedException;
    // 앨범 커버 이미지 생성
    byte[] createCover(String coverRequest) throws IOException, InterruptedException;
    // 앨범 커버 이미지 저장
    String saveCover(MultipartFileReq image);
    // 원본 음악 파일 저장
    String saveMIDI(MultipartFileReq image);
    // 믹싱 음악 파일 저장
    String saveMusic(MultipartFileReq image);

}
