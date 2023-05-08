package com.a504.chatcomposer.music.dto.response;

import static com.a504.chatcomposer.global.util.Utils.*;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

import org.springframework.format.annotation.DateTimeFormat;
import org.springframework.util.CollectionUtils;

import com.a504.chatcomposer.member.entity.FavoriteMusic;
import com.a504.chatcomposer.music.dto.Member;
import com.a504.chatcomposer.music.dto.Prompt;
import com.a504.chatcomposer.music.dto.Track;
import com.a504.chatcomposer.music.dto.enums.Beat;
import com.a504.chatcomposer.music.dto.enums.Genre;
import com.a504.chatcomposer.music.entity.Music;
import com.fasterxml.jackson.databind.PropertyNamingStrategies;
import com.fasterxml.jackson.databind.annotation.JsonNaming;
import com.querydsl.core.annotations.QueryProjection;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import lombok.extern.slf4j.Slf4j;

@Schema(description = "음악 상세 DTO")
@Getter
@Setter
@Builder
@NoArgsConstructor
@AllArgsConstructor
@JsonNaming(value = PropertyNamingStrategies.SnakeCaseStrategy.class)
@Slf4j
public class MusicDetailResp {

	@Schema(description = "음악 pk")
	private long musicId;

	@Schema(description = "작성자 정보")
	private Member member;

	@Schema(description = "음악 제목")
	private String title;

	@Schema(description = "태그 목록")
	private List<String> tags;

	@Schema(description = "음악 설명")
	private String description;

	@Schema(description = "장르")
	private Genre genre;

	@Schema(description = "박자")
	private Beat beat;

	@DateTimeFormat(pattern = "")
	@Schema(description = "음악 생성일")
	private LocalDateTime createdAt;

	@Schema(description = "트랙 목록")
	private List<Track> tracks;

	@Schema(description = "프롬프트 목록")
	private List<Prompt> prompts;

	@Schema(description = "MIDI wav 파일 S3 URL")
	private String musicSource;

	@Schema(description = "리퓨전 요청 프롬프트")
	private String mixedMusicRequest;

	@Schema(description = "리퓨전 음악 wav 파일의 S3 URL")
	private String mixedMusicSource;

	@Schema(description = "앨범 커버 요청 내용")
	private String coverRequest;

	@Schema(description = "앨범 커버 이미지의 S3 URL")
	private String coverSource;

	@Schema(description = "좋아요 수", defaultValue = "0")
	private int favoriteCount;

	@Schema(description = "좋아요 여부", defaultValue = "n", allowableValues = {"y", "n"})
	private String isMyFavorite;

	@QueryProjection
	public MusicDetailResp(Long loginUserId, Music music, Long memberId, String nickname) {

		this.musicId = music.getId();
		this.title = music.getTitle();
		this.description = music.getDescription();
		this.genre = music.getGenre();
		this.beat = music.getBeat();
		this.createdAt = music.getCreatedAt();
		this.musicSource = music.getMusicSource();
		this.mixedMusicRequest = music.getMixedMusicRequest();
		this.mixedMusicSource = music.getMixedMusicSource();
		this.coverRequest = music.getCoverRequest();
		this.coverSource = music.getCoverSource();
		this.favoriteCount = music.getFavoriteCount();

		// 작성자 정보
		this.member = Member.builder()
			.memberId(memberId)
			.nickname(nickname)
			.build();

		// music tags -> tag name 리스트로 만들기
		this.tags = music.getMusicTags().stream()
			.map(musicTag -> musicTag.getTag().getTagName())
			.collect(Collectors.toList());

		List<Track> tracks = new ArrayList<>();
		List<Prompt> prompts = new ArrayList<>();
		for (int i = 0; i < music.getTracks().size(); i++) {

			// Track Entity 리스트 -> Track DTO 리스트
			com.a504.chatcomposer.music.entity.Track track = music.getTracks().get(i);
			tracks.add(Track.builder()
				.trackId(track.getId())
				.midiDescription(track.getMidiDescription())
				.musicalInstrument(track.getMusicalInstrument())
				.build()
			);
			// Prompt Entity 리스트 -> Prompt DTO 리스트
			com.a504.chatcomposer.music.entity.Prompt prompt = track.getPrompt();
			prompts.add(Prompt.builder()
				.promptId(prompt.getId())
				.requestDescription(prompt.getRequestDescription())
				.responseDescription(prompt.getMidiDescription())
				.transferDate(prompt.getCreatedAt())
				.build());
		}
		this.tracks = tracks;
		this.prompts = prompts;

		// 내가 좋아요 한 음악인지 여부
		String isMyFavorite = NO;
		List<FavoriteMusic> favoriteMusics = music.getFavoriteMusics();
		// 누군가 좋아요 한 음악이고 로그인 된 요청이라면 음악 좋아요 여부 판단
		if (!CollectionUtils.isEmpty(favoriteMusics) && loginUserId != NOT_LOGIN) {
			for (int i = 0; i < favoriteMusics.size(); i++) {
				if (favoriteMusics.get(i).getMember().getId().equals(loginUserId)) {
					isMyFavorite = YES;
					break;
				}
			}
		}
		// 아무도 좋아요를 누르지 않은 음악이거나 로그인 하지 않은 상태의 요청은 무조건 NO
		this.isMyFavorite = isMyFavorite;

	}
}
