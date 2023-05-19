package com.a504.chatcomposer.music.dto.request;

import java.util.List;

import com.a504.chatcomposer.music.dto.PromptDetails;
import com.a504.chatcomposer.music.dto.TrackDetails;
import com.fasterxml.jackson.databind.PropertyNamingStrategies;
import com.fasterxml.jackson.databind.annotation.JsonNaming;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import lombok.ToString;

@Schema(description = "음악 저장 요청 정보 DTO")
@Getter
@Setter
@Builder
@ToString
@NoArgsConstructor
@AllArgsConstructor
@JsonNaming(value = PropertyNamingStrategies.SnakeCaseStrategy.class)
public class CompleteMusicReq {

	@Schema(description = "음악 제목")
	private String title;

	@Schema(description = "태그 목록")
	private List<String> tags;

	@Schema(description = "음악 설명")
	private String description;

	@Schema(description = "장르")
	private Integer genre;

	@Schema(description = "박자")
	private String beat;

	@Schema(description = "트랙 목록")
	private List<TrackDetails> tracks;

	@Schema(description = "프롬프트 목록")
	private List<PromptDetails> prompts;

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
}
