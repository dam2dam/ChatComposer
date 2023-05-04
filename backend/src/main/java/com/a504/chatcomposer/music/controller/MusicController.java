package com.a504.chatcomposer.music.controller;

import java.util.List;

import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import com.a504.chatcomposer.global.util.BaseResponseBody;
import com.a504.chatcomposer.music.dto.response.MusicsResp;
import com.a504.chatcomposer.music.service.FavoriteMusicService;
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
	private final FavoriteMusicService favoriteMusicService;

	@Operation(summary = "음악 리스트 조회", description = "필터 조건에 맞는 음악 리스트를 조회합니다.")
	@ApiResponses(value = {
		@ApiResponse(responseCode = "200", description = "음악 리스트 조회 완료"),
		@ApiResponse(responseCode = "404", description = "음악 리스트 조회 실패")
	})
	@GetMapping
	public ResponseEntity<?> getMusicList(
		@RequestParam(required = false, value = "genre") int genre,
		@RequestParam(required = false, value = "tag") String tag,
		@RequestParam(required = false, value = "nickname") String nickname,
		@RequestParam(required = false, value = "title") String title,
		@RequestParam(required = false, value = "is-my-favorite") String isMyFavorite) {

		// TODO: 로그인 유저 정보 (member pk) 사용
		Long loginUserId = 1L;

		List<MusicsResp> musicsResps = musicService.getMusicList(genre, tag, nickname, title, isMyFavorite,
			loginUserId);
		return ResponseEntity.ok().body(musicsResps);
	}

	@Operation(summary = "음악 좋아요 추가", description = "음악에 좋아요를 추가합니다.")
	@ApiResponses(value = {
		@ApiResponse(responseCode = "200", description = "음악 좋아요 완료"),
		@ApiResponse(responseCode = "404", description = "음악 좋아요 실패")
	})
	@PostMapping("/musics/{music_id}")
	public ResponseEntity<?> createFavoriteMusic(@PathVariable("music_id") Long musicId) {

		// TODO: 로그인 유저 정보 (member pk) 사용
		Long loginUserId = 1L;

		favoriteMusicService.createFavoriteMusic(musicId, loginUserId);
		return ResponseEntity.ok().body(new BaseResponseBody(200, "음악 좋아요를 추가했습니다."));
	}
}
