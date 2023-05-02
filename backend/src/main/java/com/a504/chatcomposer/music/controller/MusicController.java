package com.a504.chatcomposer.music.controller;

import java.util.List;

import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import com.a504.chatcomposer.music.dto.response.MusicsResp;
import com.a504.chatcomposer.music.service.MusicService;

import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.responses.ApiResponse;
import io.swagger.v3.oas.annotations.responses.ApiResponses;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;

@Tag(name = "MusicController", description = "MusicController API Document")
@RestController
@RequestMapping("/musics")
@RequiredArgsConstructor
public class MusicController {

	private final MusicService musicService;

	@Operation(summary = "음악 리스트 조회", description = "필터 조건에 맞는 음악 리스트를 조회합니다.")
	@ApiResponses(value = {
		@ApiResponse(responseCode = "200", description = "음악 리스트 조회 완료"),
		@ApiResponse(responseCode = "404", description = "음악 리스트 조회 실패")
	})
	@GetMapping
	public ResponseEntity<?> getMusicList(
		@RequestParam(required = false, value = "genre") String genre,
		@RequestParam(required = false, value = "tags") long tags,
		@RequestParam(required = false, value = "member") long memberId,
		@RequestParam(required = false, value = "title") String title,
		@RequestParam(required = false, value = "is-liked") String isLiked
	) {

		List<MusicsResp> musicsResps = musicService.getMusicList(genre, tags, memberId, title, isLiked);
		return new ResponseEntity<>(musicsResps, HttpStatus.OK);
	}
}
