package com.a504.chatcomposer.global.util;

import com.amazonaws.services.s3.AmazonS3Client;
import com.amazonaws.services.s3.model.CannedAccessControlList;
import com.amazonaws.services.s3.model.PutObjectRequest;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.time.LocalDate;
import java.util.Objects;
import java.util.Optional;

@Slf4j
@RequiredArgsConstructor
@Service
public class S3Uploader {

    private final Logger LOGGER = LoggerFactory.getLogger(S3Uploader.class);
    private final AmazonS3Client amazonS3Client;

    @Value("${cloud.aws.s3.bucket}")
    private String S3Bucket;

    public String upload(MultipartFile multipartFile, String dirName) throws IOException {
        LOGGER.info("=================upload multipartFile===========================");
        File uploadFile = convert(multipartFile).orElseThrow(() -> new IllegalArgumentException("error : MultipartFile -> file convert fail"));
        return upload(uploadFile, dirName);
    }

    // S3로 파일 업로드하기
    public String upload(File uploadFile, String dirName) {
        LOGGER.info("=================upload file===========================");
        String fileName = dirName + "/" + Math.random() * 1000 + "_" + LocalDate.now() + uploadFile.getName();
        String uploadImageUrl = putS3(uploadFile, fileName);
        removeNewFile(uploadFile); // 로컬에 생성된 File 삭제 (MultipartFile -> File 전환 하며 로컬에 파일 생성됨)

        return uploadImageUrl; // 업로드 된 파일의 S3 URL 주소 반환
    }

    // 로컬에 저장된 이미지 지우기
    private void removeNewFile(File targetFile) {
        LOGGER.info("=================removeNewFile===========================");
        if (targetFile.delete()) {
            log.info("로컬 파일이 삭제되었습니다.");
        } else {
            log.info("로컬 파일이 삭제되지 못했습니다.");
        }
    }

    // S3로 업로드
    private String putS3(File uploadFile, String fileName) {
        LOGGER.info("=================putS3===========================");
        amazonS3Client.putObject(new PutObjectRequest(S3Bucket, fileName, uploadFile).withCannedAcl(
                CannedAccessControlList.PublicRead));
        return amazonS3Client.getUrl(S3Bucket, fileName).toString();
    }

    // 로컬에 파일 업로드
    private Optional<File> convert(MultipartFile file) throws IOException {
        LOGGER.info("=================convert file===========================");
        File convertFile = new File(Objects.requireNonNull(file.getOriginalFilename()));
        if (convertFile.createNewFile()) {
            try (FileOutputStream fos = new FileOutputStream(convertFile)) {
                fos.write(file.getBytes());
            }
            return Optional.of(convertFile);
        }

        return Optional.empty();
    }

}
