package com.a504.chatcomposer.music.service;

import java.util.List;

import org.springframework.stereotype.Service;

import com.a504.chatcomposer.music.dto.response.MusicsResp;
import com.a504.chatcomposer.music.repository.MusicRepository;

import lombok.RequiredArgsConstructor;

@Service
@RequiredArgsConstructor
public class MusicService {

	private final MusicRepository musicRepository;

	/**
	 *	음악 리스트 조회
	 */
	public List<MusicsResp> getMusicList(String genre, long tags, String nickname, String title, String isLiked) {

		return musicRepository.getMusicList(genre, tags, nickname, title, isLiked);
	}
}
