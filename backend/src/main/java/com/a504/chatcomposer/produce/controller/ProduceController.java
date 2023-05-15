package com.a504.chatcomposer.produce.controller;

import com.a504.chatcomposer.music.dto.enums.Genre;
import com.a504.chatcomposer.produce.dto.request.MultipartFileReq;
import com.a504.chatcomposer.produce.dto.request.OriginalMusicReq;
import com.a504.chatcomposer.produce.dto.response.FileResp;
import com.a504.chatcomposer.produce.dto.response.FileUrlResp;
import com.a504.chatcomposer.produce.service.ProduceService;
import io.github.flashvayne.chatgpt.dto.chat.MultiChatMessage;
import io.github.flashvayne.chatgpt.service.ChatgptService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.media.Content;
import io.swagger.v3.oas.annotations.responses.ApiResponse;
import io.swagger.v3.oas.annotations.responses.ApiResponses;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@RestController
@RequiredArgsConstructor
@Tag(name = "ProduceController", description = "ProduceController API Document")
@RequestMapping("/produce")
public class ProduceController {
    private final ProduceService produceService;
    private final ChatgptService chatgptService;

    @Operation(summary = "원본 음악 저장", description = "응답받은 음악 midi 파일을 S3에 저장합니다.")
    @ApiResponses(value = {
            @ApiResponse(responseCode = "200", description = "원본 음악 저장 완료"),
            @ApiResponse(responseCode = "404", description = "원본 음악 저장 실패")
    })
    @PostMapping(path = "/musics/original", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
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
        StringBuilder moods = new StringBuilder();
        for (String value : originalMusicReq.getMoods()) {
            moods.append(value).append(", ");
        }

        // ChatGPT 프롬프트 생성
        List<MultiChatMessage> messages = Arrays.asList(
                new MultiChatMessage("system","I'm writing prompt for music generation ai.\n" +
                        "I used captions like this:\n" +
                        "1. someone is playing a high pitched melody on a steel drum. The file is of poor audio-quality.\n" +
                        "2. This is a glitch music piece. There is a synth sound rising in pitch that resembles a triangle wave. There are granular synth samples being played randomly. A virtual percussive low-to-mid bell sound is playing a melody that resembles a marimba. There is an eerie feeling of flow. This piece could be used in the soundtracks of dystopian sci-fi movies. It could also be used in exploration sequences of video games.\n" +
                        "3. This file contains an orchestral composition rising up while a lot of digital clicking sounds are in the foreground. This is an amateur recording. And the sounds seem to come from a different source. This song may be playing in an adventure videogame."),
                new MultiChatMessage("user","Now I want to make "+ Genre.findByNumber(originalMusicReq.getGenre()).name() +" music with moods contains " + moods + "and contains user's inputs as" + originalMusicReq.getRiffusionPrompt()+".\n" +
                        "Write a propmt in styles similar to above captions in one sentence of three lines."),
                new MultiChatMessage("assistant","Create a serene dance atmosphere with a dreamy melody, soothing synths, and a pulsing beat that gently propels listeners into a state of blissful tranquility, perfect for unwinding after a long day or enjoying a moment of peaceful dance."));
        String responseMessage = chatgptService.multiChat(messages);

        // 믹싱 음악 생성하기
        byte[] result = produceService.createMusic(originalMusicReq.getMusicSource(),responseMessage);

        return ResponseEntity.status(200).body(FileResp.of("리퓨젼 음악을 생성했습니다.", 200, result));
    }

    @Operation(summary = "믹싱 음악 저장", description = "응답받은 믹싱 음악 파일을 S3에 저장합니다.")
    @ApiResponses(value = {
            @ApiResponse(responseCode = "200", description = "믹싱 음악 저장 완료"),
            @ApiResponse(responseCode = "404", description = "믹싱 음악 저장 실패")
    })
    @PostMapping(path = "/musics/mixed", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public ResponseEntity<FileUrlResp> saveMusic(@ModelAttribute MultipartFileReq fileReq) {

        String source = produceService.saveMusic(fileReq);

        return ResponseEntity.status(200).body(FileUrlResp.of("믹싱 음악을 저장했습니다.", 200, source));
    }

    @Operation(summary = "앨범 커버 제작", description = "응답받은 Prompt로 앨범 커버를 제작합니다.")
    @ApiResponses(value = {
            @ApiResponse(responseCode = "200", description = "앨범 커버 제작 완료"),
            @ApiResponse(responseCode = "404", description = "앨범 커버 제작 실패")
    })
    @GetMapping(path = "/cover")
    public ResponseEntity<?> createCover(@RequestParam("cover-request") String coverRequest) throws IOException, InterruptedException {
        // cover생성파일 byte[]로 가져오기
        Map<String, Object> resultMap = new HashMap<>();
        resultMap.put("cover",produceService.createCover(coverRequest));

        // header 설정
//        HttpHeaders headers = new HttpHeaders();
//        headers.setContentType(MediaType.IMAGE_PNG);

        return ResponseEntity.status(200).body(resultMap);
    }


    @Operation(summary = "앨범 커버 저장", description = "응답받은 이미지 파일로 앨범 커버를 S3에 저장합니다.")
    @ApiResponses(value = {
            @ApiResponse(responseCode = "200", description = "앨범 커버 저장 완료"),
            @ApiResponse(responseCode = "404", description = "앨범 커버 저장 실패")
    })
    @PostMapping(path = "/cover", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public ResponseEntity<FileUrlResp> saveCover(@ModelAttribute MultipartFileReq image) {

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
