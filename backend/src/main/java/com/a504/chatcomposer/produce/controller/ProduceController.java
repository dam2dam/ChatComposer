package com.a504.chatcomposer.produce.controller;

import com.a504.chatcomposer.global.util.BaseResponseBody;
import com.a504.chatcomposer.produce.dto.request.MultipartFileReq;
import com.a504.chatcomposer.produce.dto.response.CoverUrlResp;
import com.a504.chatcomposer.produce.service.ProduceService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.media.Content;
import io.swagger.v3.oas.annotations.responses.ApiResponse;
import io.swagger.v3.oas.annotations.responses.ApiResponses;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import org.apache.http.HttpEntity;
import org.apache.http.HttpResponse;
import org.apache.http.client.methods.HttpGet;
import org.apache.http.impl.client.CloseableHttpClient;
import org.apache.http.impl.client.HttpClients;
import org.apache.http.util.EntityUtils;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.io.IOException;

@RestController
@RequiredArgsConstructor
@Tag(name = "ProduceController", description = "ProduceController API Document")
@RequestMapping("/produce")
public class ProduceController {

    private final ProduceService produceService;

    @ApiResponses(value = {
            @ApiResponse(responseCode = "200", description = "앨범 커버 제작 완료",
                    content = @Content(mediaType = "image/png") ),
            @ApiResponse(responseCode = "404", description = "앨범 커버 제작 실패",
                    content = @Content) })
    @Operation(summary = "앨범 커버 제작", description = "응답받은 Prompt로 앨범 커버를 제작합니다.")
    @GetMapping(path = "/cover")
    public ResponseEntity<?> createCover(@RequestParam("cover-request") String coverRequest) {
        // String url = "https://j8a801.p.ssafy.io/disease/predict";
        String url = "http://127.0.0.1:8885/api/generate?prompt=" + coverRequest;

        // code to send the value to FastAPI and receive an image
        try (CloseableHttpClient httpClient = HttpClients.createDefault()) {
            HttpGet request = new HttpGet(url);

            HttpResponse response = httpClient.execute(request);
            HttpEntity entity = response.getEntity();
            byte[] imageBytes = EntityUtils.toByteArray(entity);

            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.IMAGE_PNG);
            headers.setContentLength(imageBytes.length);
            return new ResponseEntity<>(imageBytes, headers, HttpStatus.OK);

        } catch (IOException e) {
            // handle exception
            throw new IllegalArgumentException("앨범 커버 생성에 실패했습니다.");
        }
    }

    @ApiResponses(value = {
            @ApiResponse(responseCode = "200", description = "앨범 커버 저장 완료",
                    content = @Content ),
            @ApiResponse(responseCode = "404", description = "앨범 커버 저장 실패",
                    content = @Content) })
    @Operation(summary = "앨범 커버 저장", description = "응답받은 이미지 파일로 앨범 커버를 S3에 저장합니다.")
    @PostMapping(path = "/cover")
    public ResponseEntity<CoverUrlResp> saveCover(@ModelAttribute("cover") MultipartFileReq multipartFileReq) {
        String coverSource = produceService.saveCover(multipartFileReq);

        return ResponseEntity.status(200).body(CoverUrlResp.of("앨범 커버를 저장했습니다.", coverSource));
    }
}
