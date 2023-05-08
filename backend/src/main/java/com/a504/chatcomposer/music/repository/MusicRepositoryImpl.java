package com.a504.chatcomposer.music.repository;

import java.util.List;

import org.springframework.util.StringUtils;

import com.a504.chatcomposer.member.entity.QFavoriteMusic;
import com.a504.chatcomposer.music.dto.enums.Genre;
import com.a504.chatcomposer.music.dto.response.MusicsResp;
import com.a504.chatcomposer.music.dto.response.QMusicsResp;
import com.a504.chatcomposer.music.entity.QMusic;
import com.a504.chatcomposer.music.entity.QMusicTag;
import com.a504.chatcomposer.music.entity.QTag;
import com.querydsl.core.types.dsl.BooleanExpression;
import com.querydsl.core.types.dsl.Expressions;
import com.querydsl.jpa.impl.JPAQueryFactory;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;

/**
 * Music 관련 디비 쿼리를 구현한 클래스 (Querydsl 사용)
 */
@RequiredArgsConstructor
@Slf4j
public class MusicRepositoryImpl implements MusicCustomRepository {

	private static final String YES = "y";
	private final JPAQueryFactory jpaQueryFactory;
	QFavoriteMusic favoriteMusic = QFavoriteMusic.favoriteMusic;
	QMusic music = QMusic.music;
	QMusicTag musicTag = QMusicTag.musicTag;
	QTag tag = QTag.tag;

	@Override
	public List<MusicsResp> getMusicList(int genre, String tagName, String nickname, String title, String isMyFavorite,
		Long loginUserId) {

		List<MusicsResp> musicsResps = jpaQueryFactory
			.select(new QMusicsResp(
				Expressions.asNumber(loginUserId).as("loginUserId"),
				music,
				music.member.id,
				music.member.memberProfile.nickname))
			.from(music)
			.leftJoin(music.musicTags, musicTag)
			.leftJoin(musicTag.tag, tag)
			.leftJoin(music.favoriteMusics, favoriteMusic)
			.where(
				genreEq(genre),
				tagNameEq(tagName),
				nicknameContains(nickname),
				titleContains(title),
				// TODO: '나'의 선호음악이므로 로그인 회원 정보를 사용해야 함
				isLikedEq(isMyFavorite, loginUserId)
			).fetch();

		log.info("MusicRepositoryImpl | getMusicList() musicsResps : {}", musicsResps);

		// return !CollectionUtils.isEmpty(musicsResps) ? musicsResps : Collections.emptyList();
		return musicsResps;
	}

	/**
	 * @return music.genre=?, or null
	 */
	private BooleanExpression genreEq(Integer genre) {
		return genre != null ? music.genre.eq(Genre.findByNumber(genre)) : null;
	}

	/**
	 * @return tag.tag_name=?, or null
	 */
	private BooleanExpression tagNameEq(String tagName) {
		return tagName != null ? tag.tagName.eq(tagName) : null;
	}

	/**
	 * @return member_profile.nickname like ? escape '!', or null
	 */
	private BooleanExpression nicknameContains(String nickname) {
		return StringUtils.hasText(nickname) ? music.member.memberProfile.nickname.contains(nickname) : null;
	}

	/**
	 * @return music.title like ? escape '!', or null
	 */
	private BooleanExpression titleContains(String title) {
		return StringUtils.hasText(title) ? music.title.contains(title) : null;
	}

	/**
	 * @return favorite_music.member_id=?, or null
	 */
	private BooleanExpression isLikedEq(String isMyFavorite, Long loginUserId) {
		return StringUtils.hasText(isMyFavorite) && isMyFavorite.equals(YES) ? favoriteMusic.member.id.eq(loginUserId) :
			null;
	}
}
