package com.a504.chatcomposer.produce.controller;

import com.a504.chatcomposer.produce.dto.request.MultipartFileReq;
import com.a504.chatcomposer.produce.dto.request.OriginalMusicReq;
import com.a504.chatcomposer.produce.dto.response.FileResp;
import com.a504.chatcomposer.produce.dto.response.FileUrlResp;
import com.a504.chatcomposer.produce.service.ProduceService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.media.Content;
import io.swagger.v3.oas.annotations.responses.ApiResponse;
import io.swagger.v3.oas.annotations.responses.ApiResponses;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;

@RestController
@RequiredArgsConstructor
@Tag(name = "ProduceController", description = "ProduceController API Document")
@RequestMapping("/produce")
public class ProduceController {
    private final ProduceService produceService;

    @Operation(summary = "원본 음악 저장", description = "응답받은 음악 midi 파일을 S3에 저장합니다.")
    @ApiResponses(value = {
            @ApiResponse(responseCode = "200", description = "원본 음악 저장 완료"),
            @ApiResponse(responseCode = "404", description = "원본 음악 저장 실패")
    })
    @PostMapping(path = "/musics/original")
    public ResponseEntity<FileUrlResp> saveMidiFile(@ModelAttribute MultipartFileReq fileReq) {

        String source = produceService.saveMIDI(fileReq);

        return ResponseEntity.status(200).body(FileUrlResp.of("원본 음악을 저장했습니다.", 200, source));
    }

    @Operation(summary = "믹싱 음악 제작", description = "음악 제작에 필요한 정보들을 가지고 riffusion 모델을 활용해 음악을 만들어준다.")
    @ApiResponses(value = {
            @ApiResponse(responseCode = "200", description = "믹싱 음악 제작 완료"),
            @ApiResponse(responseCode = "404", description = "믹싱 음악 제작 실패")
    })
    @PostMapping(path = "/musics/riffusion")
    public ResponseEntity<FileResp> createMusic(@RequestBody OriginalMusicReq originalMusicReq) throws IOException, InterruptedException {

        byte[] result = produceService.createMusic(originalMusicReq);

        return ResponseEntity.status(200).body(FileResp.of("리퓨젼 음악을 생성했습니다.", 200, result));
    }

    @Operation(summary = "앨범 커버 제작", description = "응답받은 Prompt로 앨범 커버를 제작합니다.")
    @ApiResponses(value = {
            @ApiResponse(responseCode = "200", description = "앨범 커버 제작 완료"),
            @ApiResponse(responseCode = "404", description = "앨범 커버 제작 실패")
    })
    @GetMapping(path = "/cover")
    public ResponseEntity<FileUrlResp> createCover(@RequestParam("cover-request") String coverRequest) throws IOException, InterruptedException {

        return ResponseEntity.status(200).body(produceService.createCover(coverRequest));
    }


    @Operation(summary = "앨범 커버 저장", description = "응답받은 이미지 파일로 앨범 커버를 S3에 저장합니다.")
    @ApiResponses(value = {
            @ApiResponse(responseCode = "200", description = "앨범 커버 저장 완료"),
            @ApiResponse(responseCode = "404", description = "앨범 커버 저장 실패")
    })
    @PostMapping(path = "/cover", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public ResponseEntity<FileUrlResp> saveCover(
            @ModelAttribute MultipartFileReq image) {

        String coverSource = produceService.saveCover(image);

        return ResponseEntity.status(200).body(FileUrlResp.of("앨범 커버를 저장했습니다.", 200, coverSource));
    }

//    @ApiResponses(value = {
//            @ApiResponse(responseCode = "200", description = "앨범 커버 저장 완료",
//                    content = @Content ),
//            @ApiResponse(responseCode = "404", description = "앨범 커버 저장 실패",
//                    content = @Content) })
//    @Operation(summary = "앨범 커버 저장", description = "응답받은 이미지 파일로 앨범 커버를 S3에 저장합니다.")
//    @PostMapping(path = "/cover")
//    public ResponseEntity<CoverUrlResp> saveCover(@ModelAttribute("cover") MultipartFileReq multipartFileReq) {
//        String coverSource = produceService.saveCover(multipartFileReq);
//
//        return ResponseEntity.status(200).body(CoverUrlResp.of("앨범 커버를 저장했습니다.", coverSource));
//    }

    //    @Operation(summary = "앨범 커버 제작", description = "응답받은 Prompt로 앨범 커버를 제작합니다.")
//    @GetMapping(path = "/cover")
//    public ResponseEntity<?> createCover(@RequestParam("cover-request") String coverRequest) {
//        byte[] imageBytes = produceService.createCover(coverRequest);
//
//        HttpHeaders headers = new HttpHeaders();
//        headers.setContentType(MediaType.IMAGE_PNG);
//        headers.setContentLength(imageBytes.length);
//
//        return new ResponseEntity<>(imageBytes, headers, HttpStatus.OK);
//    }
}
