package com.a504.chatcomposer.music.service;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import com.a504.chatcomposer.global.exception.CustomException;
import com.a504.chatcomposer.global.exception.CustomExceptionType;
import com.a504.chatcomposer.music.dto.PromptDetails;
import com.a504.chatcomposer.music.dto.TrackDetails;
import com.a504.chatcomposer.music.dto.enums.Beat;
import com.a504.chatcomposer.music.dto.enums.Genre;
import com.a504.chatcomposer.music.dto.request.CompleteMusicReq;
import com.a504.chatcomposer.music.dto.response.MusicDetailResp;
import com.a504.chatcomposer.music.dto.response.MusicsResp;
import com.a504.chatcomposer.music.entity.Music;
import com.a504.chatcomposer.music.entity.MusicTag;
import com.a504.chatcomposer.music.entity.Prompt;
import com.a504.chatcomposer.music.entity.Track;
import com.a504.chatcomposer.music.repository.MusicRepository;
import com.a504.chatcomposer.tag.entity.Tag;
import com.a504.chatcomposer.tag.repository.TagRepository;
import com.a504.chatcomposer.user.entity.User;
import com.a504.chatcomposer.user.repository.UserRepository;

import lombok.RequiredArgsConstructor;

@Service
@RequiredArgsConstructor
public class MusicService {

	private final MusicRepository musicRepository;
	private final UserRepository userRepository;
	private final TagRepository tagRepository;

	/**
	 * 음악 리스트 조회
	 */
	public List<MusicsResp> getMusicList(Integer genre, String tag, String nickname, String title, String isMyFavorite,
		Long loginUserId) {

		return musicRepository.getMusicList(genre, tag, nickname, title, isMyFavorite, loginUserId);
	}

	/**
	 * 음악 상세 조회
	 */
	public MusicDetailResp getMusicDetail(Long musicId, Long loginUserId) {

		return musicRepository.getMusicDetail(musicId, loginUserId);
	}

	/**
	 * 완성된 음악 저장
	 */
	@Transactional
	public String saveMusic(CompleteMusicReq completeMusicReq, Long loginUserId) {

		// User user = userRepository.findByUserId(loginUserId); // eager loading
		User user = userRepository.getReferenceById(loginUserId); // lazy loading

		Music music = Music.builder()
			.user(user)
			.title(completeMusicReq.getTitle())
			.description(completeMusicReq.getDescription())
			.genre(Genre.findByNumber(completeMusicReq.getGenre()))    // enum Genre에서 찾아서 저장
			.beat(Beat.findByBpmValue(completeMusicReq.getBeat()))    // enum Beat에서 찾아서 저장
			.musicSource(completeMusicReq.getMusicSource())
			.mixedMusicRequest(completeMusicReq.getMixedMusicRequest())
			.mixedMusicSource(completeMusicReq.getMixedMusicSource())
			.coverRequest(completeMusicReq.getCoverRequest())
			.coverSource(completeMusicReq.getCoverSource())
			.tracks(new ArrayList<>())
			.musicTags(new ArrayList<>())
			.build();

		// 모든 Tag 리스트 조회
		List<Tag> allTagList = tagRepository.findAll();
		// 모든 tagName 리스트 생성
		List<String> allTagNameList = allTagList.stream()
			.map(tag -> tag.getTagName()).collect(Collectors.toList());
		// 입력받은 태그 리스트
		List<String> inputTags = completeMusicReq.getTags();

		MusicTag musicTag = null;
		for (int i = 0; i < inputTags.size(); i++) {
			// 입력받은 태그가 등록된 태그라면 뮤직 태그 저장
			if (allTagNameList.contains(inputTags.get(i))) {
				int tagIndex = allTagNameList.indexOf(inputTags.get(i));

				musicTag = new MusicTag();
				musicTag.setMusic(music);
				musicTag.setTag(allTagList.get(tagIndex));
			} else {
				// 입력받은 태그가 등록되지 않은 태그라면 예외처리
				throw new CustomException(CustomExceptionType.UNREGISTERED_TAG);
			}
		}

		Track track = null;
		Prompt prompt = null;
		for (int i = 0; i < completeMusicReq.getTracks().size(); i++) {
			TrackDetails trackDetails = completeMusicReq.getTracks().get(i);
			PromptDetails promptDetails = completeMusicReq.getPrompts().get(i);

			track = Track.builder()
				.midiDescription(trackDetails.getMidiDescription())
				.musicalInstrument(trackDetails.getMusicalInstrument())
				.build();

			prompt = Prompt.builder()
				.requestDescription(promptDetails.getRequestDescription())
				.midiDescription(promptDetails.getResponseDescription())
				.createdAt(promptDetails.getTransferDate())
				.build();

			// 연관관계 저장
			prompt.setTrack(track);
			track.setMusic(music);
		}
		musicRepository.save(music);

		return music.getMixedMusicSource();
	}
}
