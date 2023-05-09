package com.a504.chatcomposer.music.dto.request;

import java.util.List;

import org.springframework.web.multipart.MultipartFile;

import com.a504.chatcomposer.music.dto.Prompt;
import com.a504.chatcomposer.music.dto.Track;
import com.a504.chatcomposer.music.dto.enums.Beat;
import com.a504.chatcomposer.music.dto.enums.Genre;
import com.fasterxml.jackson.databind.PropertyNamingStrategies;
import com.fasterxml.jackson.databind.annotation.JsonNaming;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

@Schema(description = "음악 저장 요청 정보 DTO")
@Getter
@Setter
@Builder
@NoArgsConstructor
@AllArgsConstructor
@JsonNaming(value = PropertyNamingStrategies.SnakeCaseStrategy.class)
public class CompleteMusicReq {

	@Schema(description = "음악 제목")
	private MultipartFile title;

	@Schema(description = "태그 목록")
	private List<String> tags;

	@Schema(description = "음악 설명")
	private String description;

	@Schema(description = "장르")
	private Genre genre;

	@Schema(description = "박자")
	private Beat beat;

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
}
