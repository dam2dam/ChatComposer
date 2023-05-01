package com.a504.chatcomposer.produce.service;

import com.a504.chatcomposer.global.util.S3Uploader;
import com.a504.chatcomposer.produce.dto.request.MultipartFileReq;
import lombok.RequiredArgsConstructor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;

@Service("ProduceService")
@RequiredArgsConstructor
@Transactional(readOnly = true)
public class ProduceServiceImpl implements ProduceService {
    private final Logger LOGGER = LoggerFactory.getLogger(ProduceServiceImpl.class);
    private final S3Uploader s3Uploader;

    @Transactional
    @Override
    public String saveCover(MultipartFileReq multipartFileReq) {
        // S3에 이미지 등록
        MultipartFile multipartFile = multipartFileReq.getImage();
        String img;
        try {
            img = s3Uploader.upload(multipartFile, "album");
        }
        catch (IOException e) {
            throw new IllegalArgumentException("앨범 커버 저장에 실패했습니다.");
        }
        LOGGER.info("================url===============\n" + img);
        return img;
    }
}
